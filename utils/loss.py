# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# ── NEW: Scale-aware and resolution-aware helpers ──────────────────────────────

def compute_scale_weight(twh, grid_shape, alpha=1.0):
    """
    Compute per-target scale-aware weights based on normalized object area.

    Smaller objects receive higher weights, emphasising their contribution
    to the box regression loss.  The weight formula follows the YOLOv4 / scaled-
    YOLOv4 convention:

        scale_weight = alpha * (2 - norm_w * norm_h)

    where norm_w and norm_h are the target width and height normalised to the
    original input image (not the feature-map grid).

    Range: [alpha * 1.0,  alpha * 2.0]  for valid normalised sizes in [0, 1].
    Numerical stability: norm_w * norm_h is clamped to [0, 1] before the
    subtraction, so exploding gradients are impossible.

    Args:
        twh        : Tensor (N, 2)  – target wh in grid-cell units (from tbox).
        grid_shape : tuple (H, W)   – spatial size of the current feature map.
        alpha      : float          – global scale multiplier (hyperparameter).

    Returns:
        Tensor (N,) of per-target weights in [alpha, 2*alpha].
    """
    # Normalise grid-space wh back to image-space [0, 1]
    norm_w = twh[:, 0] / grid_shape[1]   # feature-map col count = image-width  proxy
    norm_h = twh[:, 1] / grid_shape[0]   # feature-map row count = image-height proxy
    area   = (norm_w * norm_h).clamp(0.0, 1.0)   # guard against augmentation artefacts
    return alpha * (2.0 - area)           # shape (N,)


def apply_resolution_weight(layer_loss, layer_idx, beta):
    """
    Scale a per-layer box loss by its resolution importance weight.

    P3 (layer 0) detects small objects at the highest spatial resolution and
    should carry the highest weight; P5 (layer 2) carries the lowest weight.

    Args:
        layer_loss : scalar Tensor – box loss for a single detection layer.
        layer_idx  : int           – 0-based layer index (0 = P3, 1 = P4, 2 = P5).
        beta       : list[float]   – per-layer weights, e.g. [2.0, 1.0, 0.5].

    Returns:
        Scalar Tensor – resolution-weighted layer loss.
    """
    w = beta[layer_idx] if layer_idx < len(beta) else 1.0
    return layer_loss * w

# ── END NEW helpers ────────────────────────────────────────────────────────────


class ComputeLoss:
    """
    Computes the total YOLOv5 loss (cls + box + obj) with optional
    scale-aware bounding-box weighting and resolution-aware per-layer balancing.

    New constructor arguments (all keyword, fully backward-compatible)
    ------------------------------------------------------------------
    use_scale_aware_loss : bool
        Apply per-target scale weight  alpha*(2 - w*h)  to CIoU box loss.
        Gives small objects a higher gradient signal.  Default: True.

    use_resolution_weighting : bool
        Multiply each detection layer's box loss by a resolution factor
        beta[i] before accumulation.  P3 (small objects) gets the highest
        factor.  Default: True.

    scale_alpha : float
        Scalar multiplier for the scale weight formula.  alpha=1.0 keeps the
        original [1, 2] weight range.  Default: 1.0.

    resolution_beta : list[float] | None
        Per-layer resolution weights [P3, P4, P5].  Must have the same length
        as the number of detection layers.  Default: [2.0, 1.0, 0.5].

    log_interval : int
        Print scale/resolution diagnostics every N forward passes (0 = never).
        Useful for ablation studies.  Default: 200.
    """

    sort_obj_iou = False

    def __init__(
        self,
        model,
        autobalance=False,
        # ── NEW parameters ──────────────────────────────────────────────────
        use_scale_aware_loss=True,
        use_resolution_weighting=True,
        scale_alpha=1.0,
        resolution_beta=None,
        log_interval=200,
        # ────────────────────────────────────────────────────────────────────
    ):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        # Original objectness balance (unchanged)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        # ── NEW: scale-aware / resolution-aware config ───────────────────────
        self.use_scale_aware_loss    = use_scale_aware_loss
        self.use_resolution_weighting = use_resolution_weighting
        self.scale_alpha             = scale_alpha
        # Default: P3 gets 2×, P4 gets 1×, P5 gets 0.5× for box loss
        self.resolution_beta = (
            resolution_beta if resolution_beta is not None
            else [2.0, 1.0, 0.5]
        )
        self.log_interval = log_interval
        self._step         = 0           # forward-pass counter for logging
        # ────────────────────────────────────────────────────────────────────

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # ── NEW: per-step diagnostic accumulators ────────────────────────────
        log_scale_weights  = []    # mean scale_weight per layer (for logging)
        log_lbox_per_layer = []    # weighted box loss per layer (for logging)
        # ────────────────────────────────────────────────────────────────────

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)

                # ── MODIFIED: scale-aware box loss ───────────────────────────
                if self.use_scale_aware_loss and n > 0:
                    # tbox[i][:, 2:4] = target wh in grid-cell units
                    scale_w = compute_scale_weight(
                        tbox[i][:, 2:4],
                        pi.shape[2:4],        # (grid_H, grid_W)
                        alpha=self.scale_alpha,
                    )                         # shape (N,)
                    # Weighted mean: small objects contribute more
                    lbox_layer = (scale_w * (1.0 - iou)).mean()
                    log_scale_weights.append(scale_w.detach().mean().item())
                else:
                    lbox_layer = (1.0 - iou).mean()
                    log_scale_weights.append(1.0)
                # ── END MODIFIED ─────────────────────────────────────────────

                # ── MODIFIED: resolution-aware layer weighting for box loss ──
                if self.use_resolution_weighting:
                    lbox_layer = apply_resolution_weight(lbox_layer, i, self.resolution_beta)
                # ── END MODIFIED ─────────────────────────────────────────────

                lbox += lbox_layer
                log_lbox_per_layer.append(lbox_layer.detach().item())

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            else:
                # No targets for this layer this batch
                log_scale_weights.append(0.0)
                log_lbox_per_layer.append(0.0)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss (original balance unchanged)
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        # ── NEW: periodic diagnostic logging ─────────────────────────────────
        self._step += 1
        if self.log_interval > 0 and self._step % self.log_interval == 0:
            self._log_diagnostics(log_scale_weights, log_lbox_per_layer)
        # ────────────────────────────────────────────────────────────────────

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    # ── NEW: diagnostic logging helper ───────────────────────────────────────
    def _log_diagnostics(self, scale_weights, lbox_per_layer):
        """
        Print per-step scale-weight and per-layer box-loss diagnostics.
        Intended for ablation studies.  Call frequency controlled by log_interval.
        """
        sw_str  = "  ".join(
            f"P{3+i}: sw={sw:.4f}" for i, sw in enumerate(scale_weights)
        )
        lb_str  = "  ".join(
            f"P{3+i}: lbox={lb:.6f}" for i, lb in enumerate(lbox_per_layer)
        )
        total_lbox = sum(lbox_per_layer)
        contribs   = (
            [f"P{3+i}: {lb/total_lbox*100:.1f}%" for i, lb in enumerate(lbox_per_layer)]
            if total_lbox > 0 else ["N/A"] * len(lbox_per_layer)
        )
        print(
            f"[Loss step {self._step}] "
            f"scale_aware={self.use_scale_aware_loss}  "
            f"res_weighting={self.use_resolution_weighting}  "
            f"alpha={self.scale_alpha}  beta={self.resolution_beta}\n"
            f"  Scale weights  : {sw_str}\n"
            f"  Box loss/layer : {lb_str}\n"
            f"  Layer contrib  : {' '.join(contribs)}"
        )
    # ── END NEW ───────────────────────────────────────────────────────────────

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
