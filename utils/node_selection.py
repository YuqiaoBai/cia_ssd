import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from itertools import combinations
from tqdm import tqdm
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

def nodes_selection_triangulation(loc_ego, locs_coop):
    locs = np.array([loc_ego, *locs_coop])
    pc_range = [-40, -40, 70, 40]
    resolution = 5 #m
    grid_shape = ((pc_range[2] - pc_range[0]) // resolution, (pc_range[3] - pc_range[1]) // resolution)
    r_max = 70
    x_min, x_max = locs[:, 0].min() - r_max, locs[:, 0].max() + r_max
    y_min, y_max = locs[:, 1].min() - r_max, locs[:, 1].max() + r_max
    grid_accumulator = np.zeros(grid_shape)
    xxi, yyi = np.meshgrid(np.arange(0, grid_shape[0], 1), np.arange(0, grid_shape[1], 1), indexing='ij')
    xx, yy = np.meshgrid(np.arange(pc_range[0], pc_range[2], resolution) + resolution * 0.5,
                         np.arange(pc_range[1], pc_range[3], resolution) + resolution * 0.5, indexing='ij')


    clip_thr = 0.7
    for loc in locs:
        dists =(xx + loc_ego[0] - loc[0])**2 + (yy  + loc_ego[1]  - loc[1])**2
        info_idx = 1000 / dists
        info_idx = (np.exp(info_idx) / (np.exp(info_idx) + 1) - 0.5) * 2
        mask = np.sqrt(dists) < pc_range[2]
        info_gain = info_idx[mask]
        inds_x, inds_y = xxi[mask], yyi[mask]
        grid_accumulator[inds_x, inds_y] += info_gain
        # plot_grid_acc_and_locs(locs_coop, grid_acc, pc_range, clip_thr)


def cal_info_gain_for_comb(loc_ego, locs_coop):
    locs = np.array([loc_ego, *locs_coop])
    pc_range = [-40, -40, 70, 40]
    resolution = 5 #m
    grid_shape = ((pc_range[2] - pc_range[0]) // resolution, (pc_range[3] - pc_range[1]) // resolution)
    r_max = 70
    x_min, x_max = locs[:, 0].min() - r_max, locs[:, 0].max() + r_max
    y_min, y_max = locs[:, 1].min() - r_max, locs[:, 1].max() + r_max
    grid_accumulator = np.zeros(grid_shape)
    xxi, yyi = np.meshgrid(np.arange(0, grid_shape[0], 1), np.arange(0, grid_shape[1], 1), indexing='ij')
    xx, yy = np.meshgrid(np.arange(pc_range[0], pc_range[2], resolution) + resolution * 0.5,
                         np.arange(pc_range[1], pc_range[3], resolution) + resolution * 0.5, indexing='ij')


    clip_thr = 0.7
    info_ego = 0
    for i, loc in enumerate(locs):
        dists =(xx + loc_ego[0] - loc[0])**2 + (yy  + loc_ego[1]  - loc[1])**2
        info_idx = 1000 / dists
        info_idx = (np.exp(info_idx) / (np.exp(info_idx) + 1) - 0.5) * 2
        mask = np.sqrt(dists) < pc_range[2]
        info_gain_coop = info_idx[mask]
        inds_x, inds_y = xxi[mask], yyi[mask]
        grid_accumulator[inds_x, inds_y] += info_gain_coop
        if i==0:
            info_ego = np.clip(grid_accumulator, a_min=0, a_max=clip_thr).sum()
    grid_acc_clipped = np.clip(grid_accumulator, a_min=0, a_max=clip_thr)
    info_gain_overall = grid_acc_clipped.sum() - info_ego
    return info_gain_overall


def plot_grid_acc_and_locs(locs_coop, grid_acc, pc_range, clip_thr, resolution=5):
    ax = plt.subplot(111)
    ax.set_xlim([pc_range[0], pc_range[2]])
    ax.set_ylim([pc_range[1], pc_range[3]])
    grid_acc_clipped = np.clip(grid_acc, a_min=0, a_max=clip_thr)
    colors = grid_acc_clipped / clip_thr
    xx, yy = np.meshgrid(np.arange(pc_range[0], pc_range[2], resolution) + resolution * 0.5,
                         np.arange(pc_range[1], pc_range[3], resolution) + resolution * 0.5, indexing='ij')
    ax.scatter(xx.reshape(-1), yy.reshape(-1), s=200, c=colors.reshape(-1), marker='s', cmap='Blues')
    ax.plot(0, 0, 'r*')
    ax.plot(np.array(locs_coop)[:, 0] - loc_ego[0], np.array(locs_coop)[:, 1] - loc_ego[1], 'y*')
    plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
    plt.close()

def info_gain_of_combs(path_in, path_out):
    if os.path.exists(os.path.join(path_out, "info_gains")):
        selection_list = {}
        for file in tqdm(glob(path_out + "/info_gains/*.npy")):
            selection = []
            info_gain_dict = np.load(file, allow_pickle=True).item()
            for i in range(1, 5):
                grp_ids = info_gain_dict[i]['grp_ids']
                if len(grp_ids)>0:
                    selection.append(grp_ids[np.array(info_gain_dict[i]['info_gain']).argmax()])
                else:
                    selection.append([])
            selection_list[file] = selection
        np.save(os.path.join(path_out, 'info_gain_selection.npy'), selection_list)
    else:
        os.makedirs(os.path.join(path_out, "info_gains"))
        info_gain_dict = dict()
        selection_list = {}
        for file in tqdm(glob(path_in + "/*.npy")):
            selection = []
            for i in range(1, 5):
                info_gain_dict[i] = dict()
                tfs = np.load(file, allow_pickle=True).item()
                loc_ego = tfs.pop('tf_ego')
                loc_ego = [loc_ego[0, -1], loc_ego[1, -1]]
                locs_coop = []
                ids_coop = np.array(list(tfs.keys()))
                for id, tf in tfs.items():
                    locs_coop.append([tf[0, -1], tf[1, -1]])
                locs_coop = np.array(locs_coop)
                combs = list(combinations(np.arange(len(locs_coop)), i))
                info_gain_dict[i]['grp_ids'] = []
                info_gain_dict[i]['grp_locs'] = []
                info_gain_dict[i]['info_gain'] = []
                for comb in combs:
                    info_gain = cal_info_gain_for_comb(loc_ego, locs_coop[list(comb)])
                    info_gain_dict[i]['grp_ids'].append(ids_coop[list(comb)])
                    info_gain_dict[i]['grp_locs'].append(locs_coop[list(comb)])
                    info_gain_dict[i]['info_gain'].append(info_gain)
                grp_ids = info_gain_dict[i]['grp_ids']
                if len(grp_ids)>0:
                    selection.append(grp_ids[np.array(info_gain_dict[i]['info_gain']).argmax()])
                else:
                    selection.append([])
            selection_list[file] = selection
            np.save(file.replace('tfs', 'info_gains'), info_gain_dict)
        np.save(os.path.join(path_out, 'info_gain_selection.npy'), selection_list)


def nodes_selection_fps(path_in, path_out):
    selection_list = {}
    for file in tqdm(glob(path_in + "/*.npy")):
        selection = []
        tfs = np.load(file, allow_pickle=True).item()
        loc_ego = tfs.pop('tf_ego')
        loc_ego = [loc_ego[0, -1], loc_ego[1, -1]]
        locs_coop = []
        ids_coop = np.array(list(tfs.keys()))
        for id, tf in tfs.items():
            locs_coop.append([tf[0, -1], tf[1, -1]])
        locs_coop = np.array(locs_coop)

        for i in range(1, 5):
            solution_set = fps(loc_ego, locs_coop, i)
            ids = [] if len(solution_set)==0 else ids_coop[solution_set]
            selection.append(ids)
        selection_list[file] = selection
    np.save(os.path.join(path_out, 'fps_selection.npy'), selection_list)


def fps(loc_ego, locs_coop, n_samples=5):
    if len(locs_coop) < n_samples:
        return []
    locs = [loc_ego, *locs_coop]
    solution_set = [0]
    remaining_points = np.arange(len(locs)).tolist()
    remaining_points.remove(0)

    def distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    for _ in range(n_samples):
        distances = [distance(locs[p], locs[solution_set[0]]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(locs[p], locs[s]))
        solution_set.append(remaining_points.pop(distances.index(max(distances))) - 1)
    return solution_set[1:]

def triangle_area(a, b, c):
    s = (a + b + c) / 2 # semi-perimeter
    EPS = 1e-6
    if abs(s-a)<EPS or abs(s-b)<EPS or abs(s-c)<EPS:
        return EPS
    # if (s * (s - a) * (s - b) * (s - c)) < EPS:
    #     print("debug")
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area

def congruent_index(points):
    tri = Delaunay(points)
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
    # plt.close()
    c_index_sum = 0
    area_sum = 0
    for i in range(tri.nsimplex):
        vertices = points[tri.simplices[i]]
        a = np.linalg.norm((vertices[0] - vertices[1]))
        b = np.linalg.norm((vertices[1] - vertices[2]))
        c = np.linalg.norm((vertices[2] - vertices[0]))
        m = (a + b + c) / 3
        area = triangle_area(a, b, c)
        area_sum += area
        c_index_sum += area / triangle_area(m, m, m)
    c_index = c_index_sum / tri.nsimplex * area_sum
    return c_index

def congruent_index_v2(loc_ego, loc_coops):
    offsets = loc_coops - loc_ego
    directions_to_ego = np.arctan2(offsets[:, 1], offsets[:, 0]) + np.pi
    order = np.argsort(directions_to_ego)
    directions_to_ego = directions_to_ego[order]
    loc_coops = loc_coops[order]
    area_sum = 0
    plt.plot(loc_ego[0, 0], loc_ego[0, 1], "r*", markersize=20)
    for i, loc in enumerate(loc_coops):
        angle_offset = directions_to_ego[i] - directions_to_ego[i-1]
        if angle_offset>np.pi:
            continue
        a = np.linalg.norm(loc_coops[i-1] - loc)
        b = np.linalg.norm(loc - loc_ego)
        c = np.linalg.norm(loc_ego - loc_coops[i-1])
        area_sum += triangle_area(a, b, c)
        points = np.concatenate([loc_ego, loc_coops[[i-1, i]], loc_ego], axis=0)
        plt.plot(points[:, 0], points[:, 1])
        plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
    plt.close()
    return area_sum


def delaunay_selection(path_in, path_out):
    com_range = 40
    selection_list = {}
    itr = 0
    for file in tqdm(glob(path_in + "/*.npy")):
        selection = []
        # itr += 1
        # if itr < 1997:
        #     continue

        tfs = np.load(file, allow_pickle=True).item()
        loc_ego = tfs.pop('tf_ego')
        loc_ego = np.array([loc_ego[0, -1], loc_ego[1, -1]])
        locs_coop = []
        ids_coop = np.array(list(tfs.keys()))
        for id, tf in tfs.items():
            loc_coop = np.array([tf[0, -1], tf[1, -1]])
            if np.linalg.norm(loc_coop- loc_ego) < com_range:
                locs_coop.append(loc_coop)
        locs_coop = np.array(locs_coop)
        for i in range(1, 5):
            if i==1 and len(locs_coop) > 0:
                selection.append([ids_coop[np.linalg.norm(locs_coop - loc_ego, axis=1).argmax()]])
                continue
            elif len(locs_coop) < i:
                selection.append([])
                continue
            combs = list(combinations(np.arange(len(locs_coop)), i))
            congruence_max = 0
            comb_selected = []
            for comb in combs:
                congruence = congruent_index(np.concatenate([loc_ego.reshape(1, 2),
                                                             locs_coop[list(comb)]], axis=0))
                if congruence > congruence_max:
                    congruence_max = congruence
                    comb_selected = comb
            selection.append(ids_coop[list(comb_selected)])
            #plt.close()
        selection_list[file.rsplit("/")[-1][:-4]] = selection
    np.save(os.path.join(path_out, 'delaunay_selection_40.npy'), selection_list)

def triangle_selection(path_in, path_out):
    com_range = 40
    selection_list = {}
    itr = 0
    for file in tqdm(glob(path_in + "/*.npy")):
        selection = []
        tfs = np.load(file, allow_pickle=True).item()
        loc_ego = tfs.pop('tf_ego')
        loc_ego = np.array([loc_ego[0, -1], loc_ego[1, -1]])
        locs_coop = []
        ids_coop = np.array(list(tfs.keys()))
        for id, tf in tfs.items():
            loc_coop = np.array([tf[0, -1], tf[1, -1]])
            if np.linalg.norm(loc_coop- loc_ego) < com_range:
                locs_coop.append(loc_coop)
        locs_coop = np.array(locs_coop)
        for i in range(3, 5):
            if i==1 and len(locs_coop) > 0:
                selection.append([ids_coop[np.linalg.norm(locs_coop - loc_ego, axis=1).argmax()]])
                continue
            elif len(locs_coop) < i:
                selection.append([])
                continue
            combs = list(combinations(np.arange(len(locs_coop)), i))
            congruence_max = 0
            comb_selected = []
            for comb in combs:
                congruence = congruent_index_v2(loc_ego.reshape(1, 2), np.array(locs_coop[list(comb)]))
                if congruence > congruence_max:
                    congruence_max = congruence
                    comb_selected = comb
            selection.append(ids_coop[list(comb_selected)])
        selection_list[file.rsplit("/")[-1][:-4]] = selection
    np.save(os.path.join(path_out, 'delaunay_selection_40.npy'), selection_list)

def kmeans_selection(path_in, path_out, com_range=60):
    selection_list = {}
    itr = 0
    for file in tqdm(glob(path_in + "/*.npy")):
        selection = []
        tfs = np.load(file, allow_pickle=True).item()
        loc_ego = tfs.pop('tf_ego')
        loc_ego = np.array([loc_ego[0, -1], loc_ego[1, -1]])
        locs_coop = []
        ids_coop = np.array(list(tfs.keys()))
        for id, tf in tfs.items():
            loc_coop = np.array([tf[0, -1], tf[1, -1]])
            if np.linalg.norm(loc_coop- loc_ego) < com_range:
                locs_coop.append(loc_coop)
        locs_coop = np.array(locs_coop)
        for i in range(1, 5):
            if i==1 and len(locs_coop) > 0:
                selection.append([ids_coop[np.linalg.norm(locs_coop - loc_ego, axis=1).argmax()]])
                continue
            elif len(locs_coop) < i:
                selection.append([])
                continue
            points = np.concatenate([loc_ego.reshape(1, 2),locs_coop], axis=0)
            inds = np.arange(len(points))
            kmeans = KMeans(n_clusters=i+1, random_state=0).fit(points)
            labels = kmeans.labels_
            ego_cluster_label = labels[0]
            clusters = [inds[labels==l] for l in range(max(labels) + 1)]
            combs = combinations(np.arange(len(locs_coop)), i)
            # combs_selected = []
            congruence_max = 0
            comb_selected = []
            for comb in combs:
                comb_cluster_inds = labels[np.array(comb) + 1]
                if len(np.unique(comb_cluster_inds))==len(comb_cluster_inds) and \
                        np.all(comb_cluster_inds!=ego_cluster_label):
                    # combs_selected.append(comb)
                    congruence = congruent_index(np.concatenate([loc_ego.reshape(1, 2),
                                                                 locs_coop[list(comb)]], axis=0))
                    if congruence > congruence_max:
                        congruence_max = congruence
                        comb_selected = comb
            selection.append(ids_coop[list(comb_selected)])
            # pts = np.concatenate([loc_ego.reshape(1, 2), points[np.array(comb_selected) + 1]], axis=0)
            # tri = Delaunay(pts)
            # plt.triplot(pts[:, 0], pts[:, 1], tri.simplices)
            # plt.axis('equal')
            # plt.plot(loc_ego[0], loc_ego[1], 'k*', markersize=20)
            # for cluster in clusters:
            #     plt.plot(points[cluster, 0], points[cluster, 1], 'o', markersize=10)
            # plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
            # plt.close()
        selection_list[file.rsplit("/")[-1][:-4]] = selection
    np.save(os.path.join(path_out, 'kmeans_selection_{:d}.npy'.format(com_range)), selection_list)




def plot_selections(data_path):
    info_gain_selection = np.load(os.path.join(data_path, "info_gain_selection.npy"), allow_pickle=True).item()
    fps_selection = np.load(os.path.join(data_path, "fps_selection.npy"), allow_pickle=True).item()
    delaunay_selection = np.load(os.path.join(data_path, "delaunay_selection_40.npy"), allow_pickle=True).item()
    for i, frame in enumerate(delaunay_selection.keys()):
        file = os.path.join(data_path, 'tfs', frame + '.npy')
        tfs = np.load(file, allow_pickle=True).item()
        if len(tfs)<5:
            continue
        loc_ego = tfs.pop('tf_ego')
        loc_ego = np.array([loc_ego[0, -1], loc_ego[1, -1]])
        loc_all = np.array([[tfs[id][0, -1], tfs[id][1, -1]] for id in tfs.keys()])
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(131)
        locs = np.array([[tfs[id][0, -1], tfs[id][1, -1]] for id in \
                         info_gain_selection[file.replace('tfs', 'info_gains')][3]])
        ax = plot_points(ax, loc_ego, locs, loc_all)
        ax = plt.subplot(132)
        locs = np.array([[tfs[id][0, -1], tfs[id][1, -1]] for id in fps_selection[file][3]])
        ax = plot_points(ax, loc_ego, locs, loc_all)
        ax = plt.subplot(133)
        if len(delaunay_selection[frame][3])!=0:
            locs = np.array([[tfs[id][0, -1], tfs[id][1, -1]] for id in delaunay_selection[frame][3]])
            ax = plot_points(ax, loc_ego, locs, loc_all)
        plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
        plt.close()

def plot_points(ax, loc_ego, loc_coops, loc_all):
    ax.plot(loc_all[:, 0], loc_all[:, 1], 'go')
    ax.plot(loc_ego[0], loc_ego[1], 'r*')
    ax.plot(loc_coops[:, 0], loc_coops[:, 1], 'r*')
    return ax


if __name__=="__main__":
    path = "/media/hdd/ophelia/koko/data/synthdata_20veh_60m/tfs"
    out = "/media/hdd/ophelia/koko/data/synthdata_20veh_60m"
    # info_gain_of_combs(path, out)
    # nodes_selection_fps(path, out)
    #delaunay_selection(path, out)
    # triangle_selection(path, out)
    for r in [30, 50]:
        kmeans_selection(path, out , r)
    #plot_selections(out)




