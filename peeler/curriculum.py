"""Curriculum management for peeler training.

Handles competence-based phase advancement, LR scheduling across phases,
and curriculum sample allocation. Pool building stays in the dataset
(it owns the bucket structure).
"""
import math
from collections import deque

CURRICULUM_BUCKETS = ['2-4', '5-11', '12-26', '27-51', '52-101', '102-300']
MIN_HOLD_EPOCHS = 5
COMPETENCE_THRESHOLD = 0.7


class _PhaseScheduler:
    """Self-contained LR scheduler for a single curriculum phase.

    Owns a 3-mode state machine that computes LR directly from tracked
    state — no delegation to sub-schedulers. Guarantees continuity across
    all transitions.

    Modes:
      PHASE_RAMP    — linear interpolation from current LR to phase peak LR
      COSINE        — cosine decay from peak LR to eta_min over T_0 epochs
      RESTART_RAMP  — linear interpolation from current LR to gamma*peak LR
    """

    PHASE_RAMP = 'phase_ramp'
    COSINE = 'cosine'
    RESTART_RAMP = 'restart_ramp'

    def __init__(self, optimizer, peak_lr, eta_min, t0, restart_decay,
                 phase_ramp_epochs, restart_ramp_epochs):
        self._optimizer = optimizer
        self._peak_lr = peak_lr
        self._eta_min = eta_min
        self._t0 = t0
        self.restart_decay = restart_decay
        self._phase_ramp_epochs = max(phase_ramp_epochs, 1)
        self._restart_ramp_epochs = max(restart_ramp_epochs, 1)

        # State
        self._mode = self.COSINE
        self._cosine_epoch = 0
        self._ramp_step = 0

    def step(self):
        if self._mode == self.PHASE_RAMP:
            self._step_phase_ramp()
        elif self._mode == self.COSINE:
            self._step_cosine()
        else:
            self._step_restart_ramp()

    def advance_phase(self, new_peak_lr, new_t0):
        """Enter phase ramp mode targeting the new peak LR and cycle length."""
        self._peak_lr = new_peak_lr
        self._t0 = new_t0
        self._mode = self.PHASE_RAMP
        self._ramp_step = 0

    def _step_phase_ramp(self):
        current_lr = self._optimizer.param_groups[0]['lr']
        self._ramp_step += 1
        t = self._ramp_step / self._phase_ramp_epochs
        new_lr = current_lr + (self._peak_lr - current_lr) * t
        self._set_lr(new_lr)

        if self._ramp_step >= self._phase_ramp_epochs:
            self._mode = self.COSINE
            self._cosine_epoch = 0

    def _step_cosine(self):
        t = self._cosine_epoch / self._t0
        new_lr = self._eta_min + (self._peak_lr - self._eta_min) * (
            1 + math.cos(math.pi * t)
        ) / 2
        self._set_lr(new_lr)
        self._cosine_epoch += 1

        if self._cosine_epoch >= self._t0:
            self._mode = self.RESTART_RAMP
            self._ramp_step = 0
            self._peak_lr *= self.restart_decay

    def _step_restart_ramp(self):
        current_lr = self._optimizer.param_groups[0]['lr']
        self._ramp_step += 1
        t = self._ramp_step / self._restart_ramp_epochs
        new_lr = current_lr + (self._peak_lr - current_lr) * t
        self._set_lr(new_lr)

        if self._ramp_step >= self._restart_ramp_epochs:
            self._mode = self.COSINE
            self._cosine_epoch = 0

    def _set_lr(self, lr):
        for pg in self._optimizer.param_groups:
            pg['lr'] = lr

    @property
    def ramp_progress(self):
        if self._mode == self.PHASE_RAMP:
            return self._ramp_step / self._phase_ramp_epochs
        return 1.0

    def state_dict(self):
        return {
            'mode': self._mode,
            'peak_lr': self._peak_lr,
            'cosine_epoch': self._cosine_epoch,
            'ramp_step': self._ramp_step,
        }

    def load_state_dict(self, state_dict):
        self._mode = state_dict['mode']
        self._peak_lr = state_dict['peak_lr']
        self._cosine_epoch = state_dict['cosine_epoch']
        self._ramp_step = state_dict['ramp_step']


class CurriculumManager:
    """Top-level scheduler: competence-based phase advancement + LR scheduling.

    Manages phase ramps, restart schedulers, and all LR state internally.
    The training loop calls ``step()`` once per epoch.
    """

    def __init__(self, optimizer, eta_min, scheduler_cfg, log_callback=None):
        self._optimizer = optimizer
        self._eta_min = eta_min
        self._log_callback = log_callback
        self._initial_lr = optimizer.param_groups[0]['lr']

        # Phase config
        self._cur_phase = 0
        self._ari_window = deque(maxlen=MIN_HOLD_EPOCHS)

        # Scheduler config
        self._phase_decay = scheduler_cfg.get('phase_decay', 0.8)
        self._t0_base = scheduler_cfg.get('T_0', 50)
        self._t0_growth = scheduler_cfg.get('T_0_growth', 1.0)
        self._ramp_epochs = scheduler_cfg.get('ramp_epochs', 2)
        self.restart_decay = scheduler_cfg.get('restart_decay', 0.9)
        self._phase_ramp_epochs = scheduler_cfg.get('phase_ramp_epochs', 1)

        # Create initial phase scheduler
        self._scheduler = self._create_phase_scheduler(0)

    def _create_phase_scheduler(self, phase):
        """Create a new phase scheduler with ramp + restart."""
        peak_lr = self._initial_lr * (self._phase_decay ** phase)
        t0 = int(self._t0_base * (self._t0_growth ** phase))

        scheduler = _PhaseScheduler(
            self._optimizer,
            peak_lr=peak_lr,
            eta_min=self._eta_min,
            t0=t0,
            restart_decay=self.restart_decay,
            phase_ramp_epochs=self._phase_ramp_epochs,
            restart_ramp_epochs=self._ramp_epochs,
        )

        log_msg = (
            f'[LR] Phase {phase} Activated: Peak_LR={peak_lr:.2e}, '
            f'T_0={t0}, PhaseRamp={self._phase_ramp_epochs}ep, RestartRamp={self._ramp_epochs}ep'
        )
        if self._log_callback:
            self._log_callback(log_msg)

        return scheduler

    @property
    def phase(self):
        return self._cur_phase

    @property
    def phase_label(self):
        return CURRICULUM_BUCKETS[self._cur_phase]

    @property
    def ramp_progress(self):
        return self._scheduler.ramp_progress

    def step(self, bucket_ari_dict):
        """Report ARI, advance phase (if competent), and step the scheduler."""
        self.report_competence(bucket_ari_dict)
        self.advance_phase()
        self._scheduler.step()

    def state_dict(self):
        return {
            'phase': self._cur_phase,
            'ari_window': list(self._ari_window),
            'initial_lr': self._initial_lr,
            'scheduler': self._scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._cur_phase = state_dict['phase']
        self._ari_window = deque(state_dict['ari_window'], maxlen=MIN_HOLD_EPOCHS)
        self._initial_lr = state_dict.get('initial_lr', self._initial_lr)
        self._scheduler.load_state_dict(state_dict['scheduler'])

    def report_competence(self, bucket_ari_dict):
        """Report per-bucket training ARI to drive competence-based progression."""
        hardest_bucket = CURRICULUM_BUCKETS[self._cur_phase]
        bucket_data = bucket_ari_dict.get(hardest_bucket)
        if bucket_data is None:
            return
        train_ari = bucket_data.get('ari', 0.0)
        self._ari_window.append(train_ari)

    def advance_phase(self):
        """Check competence and advance phase if threshold met."""
        if len(self._ari_window) == MIN_HOLD_EPOCHS:
            avg_ari = sum(self._ari_window) / MIN_HOLD_EPOCHS
            if avg_ari >= COMPETENCE_THRESHOLD and self._cur_phase < len(CURRICULUM_BUCKETS) - 1:
                self._cur_phase += 1
                self._ari_window.clear()
                new_peak_lr = self._initial_lr * (self._phase_decay ** self._cur_phase)
                new_t0 = int(self._t0_base * (self._t0_growth ** self._cur_phase))
                self._scheduler.advance_phase(new_peak_lr, new_t0)
