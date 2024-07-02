import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def point_form_from_xywh(boxes):
    return np.concatenate((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]), axis=1)

def point_form(boxes):
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), axis=1)

def center_size(boxes):
    return np.concatenate(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), axis=1)

def intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        np.broadcast_to(box_a[:, 2:].reshape((A, 1, 2)), (A, B, 2)),
        np.broadcast_to(box_b[:, 2:].reshape((1, B, 2)), (A, B, 2))
    )
    min_xy = np.maximum(
        np.broadcast_to(box_a[:, :2].reshape((A, 1, 2)), (A, B, 2)),
        np.broadcast_to(box_b[:, :2].reshape((1, B, 2)), (A, B, 2))
    )
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).reshape(-1, 1).repeat(box_b.shape[0], axis=1)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).reshape(1, -1).repeat(box_a.shape[0], axis=0)
    union = area_a + area_b - inter
    return inter / union

def matrix_iou(a, b):
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)

def matrix_iof(a, b):
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    truths = point_form(truths)
    overlaps = jaccard(truths, point_form(priors))

    best_prior_overlap = np.max(overlaps, axis=1, keepdims=True)
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = np.squeeze(np.where(valid_gt_idx))
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    best_truth_overlap = np.max(overlaps, axis=0, keepdims=True)
    best_truth_idx = np.argmax(overlaps, axis=0).squeeze()
    best_truth_overlap = np.squeeze(best_truth_overlap)
    best_prior_idx = np.argmax(overlaps, axis=1).squeeze()
    best_prior_idx_filter = np.squeeze(best_prior_idx_filter)
    best_prior_overlap = np.squeeze(best_truth_overlap)

    best_truth_overlap[best_prior_idx_filter] = 2

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    return np.concatenate([g_cxcy, g_wh], axis=1)

def decode(loc, priors, variances):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def log_sum_exp(x):
    x_max = np.max(x, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=1, keepdims=True)) + x_max
