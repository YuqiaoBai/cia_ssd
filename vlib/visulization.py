from vlib.point import draw_points_boxes_wandb, draw_points_boxes_plt


def draw_points_boxes_bev_3d(points, pred_boxes, gt_boxes, pc_range):
    """
    visualize the result of the first batch, input shoulb be data from only one batch
    """
    draw_points_boxes_wandb(points, boxes_pred=pred_boxes, boxes_gt=gt_boxes)
    draw_points_boxes_plt(pc_range, points[:, :2], boxes_pred=pred_boxes, boxes_gt=gt_boxes)


def draw_points_boxes_bev(points, pred_boxes, gt_boxes, pc_range):
    """
    visualize the result of the first batch, input shoulb be data from only one batch
    """
    draw_points_boxes_plt(pc_range, points[:, :2], boxes_pred=pred_boxes, boxes_gt=gt_boxes)


def draw_points_boxes_3d(points, pred_boxes, gt_boxes, pc_range):
    """
    visualize the result of the first batch, input shoulb be data from only one batch
    """
    draw_points_boxes_wandb(points, boxes_pred=pred_boxes, boxes_gt=gt_boxes)