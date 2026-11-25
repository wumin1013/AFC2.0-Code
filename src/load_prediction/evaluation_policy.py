"""Explicit scoring scales and statistical groups, independent of prediction."""
import math


def group_key(metadata):
    """New events have their own namespace, never invented development rounds."""
    if metadata.get('group_id') is not None:
        value = metadata['group_id']
        if not isinstance(value, str) or not value.strip():
            raise ValueError('group_id必须是非空字符串')
        return ('event', value)
    if metadata.get('round_id') is None:
        raise ValueError('缺少独立组或原始轮次身份')
    return ('round', metadata['round_id'])


def scoring_parameters(bundle, explicit=None):
    values = dict(scale_w=bundle.normalization['target_scale_w'],
                  high_threshold_w=bundle.normalization['high_load_threshold_w'],
                  low_threshold_w=bundle.normalization['low_load_threshold_w']) if explicit is None else dict(explicit)
    if set(values) != {'scale_w', 'high_threshold_w', 'low_threshold_w'}:
        raise ValueError('评分参数必须完整指定尺度、高负载及低负载阈值')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
        raise ValueError('评分参数必须为有限数值')
    if values['scale_w'] <= 0 or values['low_threshold_w'] > values['high_threshold_w']:
        raise ValueError('评分尺度或阈值次序无效')
    return values
