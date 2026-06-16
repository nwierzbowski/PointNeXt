"""Phase configuration loaded from YAML. Shared between scheduler and loss."""


def load_phase_config(cfg):
    """Load and validate phase config from YAML.

    Returns dict with:
        phase_1_epochs: int (Phase 1 duration)
        phase_2_epochs: int (Phase 2 duration)
        c1_weight: float (Phase 1 contrastive weight)
        c2_weight: float (Phase 2 contrastive weight)
    """
    p = cfg.phases
    return {
        'phase_1_epochs': p['phase_1_epochs'],
        'phase_2_epochs': p['phase_2_epochs'],
        'c1_weight': p['phase_1_contrastive_weight'],
        'c2_weight': p['phase_2_contrastive_weight'],
    }
