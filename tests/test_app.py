import pytest
import pandas as pd
from src.calculator import Indicator, IndicatorConfig  # your_moduleを実際のモジュール名に置き換えてください

# テスト用のフィクスチャ
@pytest.fixture
def normal_scores() -> pd.Series:
    return pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

@pytest.fixture
def normal_flgs() -> pd.Series:
    return pd.Series([0, 1, 0, 1, 1])

@pytest.fixture
def normal_config() -> IndicatorConfig:
    return IndicatorConfig()

@pytest.fixture
def indicator(normal_scores: pd.Series, normal_flgs: pd.Series, normal_config: IndicatorConfig) -> Indicator:
    return Indicator(normal_config, normal_scores, normal_flgs)

# precision_at_kメソッドのテスト
def test_precision_at_k_normal(indicator: Indicator):
    assert indicator.precision_at_k(0.4) == 0.5
    assert indicator.precision_at_k(1.0) == 0.6

def test_precision_at_k_abnormal(indicator: Indicator):
    with pytest.raises(ValueError):
        indicator.precision_at_k(-0.1)
    with pytest.raises(ValueError):
        indicator.precision_at_k(1.1)

def test_precision_at_k_mismatch_length(indicator: Indicator):
    indicator.scores = pd.Series([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        indicator.precision_at_k(0.4)

def test_precision_at_k_empty(indicator: Indicator):
    indicator.scores = pd.Series([])
    indicator.flgs = pd.Series([])
    assert indicator.precision_at_k(0.4) == 0

# deterrent_effectメソッドのテスト
def test_deterrent_effect_normal(indicator: Indicator):
    assert indicator.deterrent_effect() == 0.6 * 10000

def test_deterrent_effect_config_change(indicator: Indicator):
    indicator.config.unit_price = 20000
    indicator.config.target_ratio = 0.2
    assert indicator.deterrent_effect() == 0.5 * 20000

def test_deterrent_effect_empty(indicator: Indicator):
    indicator.scores = pd.Series([])
    indicator.flgs = pd.Series([])
    assert indicator.deterrent_effect() == 0

# エッジケースと境界値のテスト
def test_precision_at_k_edge_cases(indicator: Indicator):
    assert indicator.precision_at_k(0) == 0
    assert indicator.precision_at_k(1) == 0.6
