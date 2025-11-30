import pytest

from src.risk.manager import RiskManager


@pytest.fixture
def risk_manager():
    return RiskManager()


@pytest.mark.parametrize(
    "position,max_position,expected",
    [(0.5, 1.0, True), (-1.0, 1.5, True), (2.0, 1.0, False), (1.0, -1.0, False)],
)
def test_position_limits(risk_manager, position, max_position, expected):
    assert risk_manager.check_position_limit(position, max_position) is expected


def test_stop_loss_calculation(risk_manager):
    assert risk_manager.calculate_stop_loss(100.0, 2.5) == pytest.approx(95.0)
    assert risk_manager.calculate_stop_loss(-1.0, 2.0) == 0.0

