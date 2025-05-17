import torch
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import os
import ssgetpy  # 用于精确搜索 SuiteSparse 图

# 从 SmartGD 框架导入必要的类
from smartgd.constants import DATASET_ROOT  # 使用常量
from smartgd.model import Generator
from smartgd.data import GraphDrawingData
from smartgd.datasets import SuiteSparseDataset  # 导入 SuiteSparseDataset
from smartgd.transformations import Compose, Center, NormalizeRotation, RescaleByStress
from networkx.algorithms.community import girvan_newman

# --- 0. 配置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# <<<--- 选择一个预训练模型检查点 ---<<<
generator_checkpoint_path = "./generator_stress_only.pt"  # 或 "./generator_xing_only.pt"

# 模型参数
generator_params = Generator.Params(
    num_blocks=11, block_depth=3, block_width=8, block_output_dim=8,
    edge_net_depth=2, edge_net_width=16, edge_attr_dim=2, node_attr_dim=2,
)

# 图的配置
# <<<--- 设置为 "suitesparse" 并指定图名 ---<<<
graph_source = "suitesparse"
# 论文 Figure 9 中使用的 SuiteSparse 图: plskz362, bfwa782, can_838, utm1700b
graph_name_to_draw = "utm1700b"  # 例如，绘制 bfwa782

# SuiteSparseDataset 的根目录
suite_sparse_dataset_root = DATASET_ROOT  # 通常是 "./datasets"

# --- 1. 加载模型 ---
print(f"Loading generator model from: {generator_checkpoint_path}")
if not os.path.exists(generator_checkpoint_path):
    print(f"Error: Generator checkpoint not found at {generator_checkpoint_path}.")
    exit()
generator = Generator(params=generator_params).to(device)
try:
    generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=device, weights_only=True))
    generator.eval()
    print("Generator model loaded successfully.")
except Exception as e:
    print(f"Error loading generator model: {e}")
    exit()

# --- 2. 准备图数据 ---
GraphDrawingData.set_optional_fields(["edge_pair_metaindex"])

nx_graph_input = None
graph_data_item_raw = None  # 用于存储从数据集中加载的原始 Data 对象
data_name_for_plot = graph_name_to_draw  # 先用目标名称

if graph_source == "suitesparse":
    print(f"Attempting to load '{graph_name_to_draw}' from SuiteSparseDataset...")
    try:
        # 步骤 2a: 使用 ssgetpy 查找特定图的信息
        # 这有助于我们知道图的节点数，以便更精确地实例化 SuiteSparseDataset
        # 或者直接修改 SuiteSparseDataset 以接受按名称加载的逻辑（更高级）
        print(f"Searching for '{graph_name_to_draw}' metadata using ssgetpy...")
        search_results = ssgetpy.search(name=graph_name_to_draw)  # 按确切名称搜索

        if not search_results:
            print(f"Error: Graph '{graph_name_to_draw}' not found via ssgetpy search.")
            print("Please ensure the graph name is correct and exists in the SuiteSparse Matrix Collection.")
            exit()

        # 假设只找到一个匹配项
        target_graph_info = search_results[0]
        print(
            f"Found graph: {target_graph_info.name}, Group: {target_graph_info.group}, Nodes: {target_graph_info.rows}, Edges: {target_graph_info.nnz}")

        # 步骤 2b: 实例化 SuiteSparseDataset
        # 我们将 min_nodes 和 max_nodes 设置为目标图的节点数，limit 设置为 1，
        # 以便 SuiteSparseDataset 只关注这个特定的图。
        # 这依赖于 SuiteSparseDataset 的 __init__ 中的 ssgetpy.search
        # 能够通过节点数范围和限制找到这个特定的图。
        # 注意：如果多个图具有完全相同的节点数，这可能仍会选择列表中的第一个。
        # 一个更鲁棒的方法是修改 SuiteSparseDataset 以接受一个预填充的 graph_list。
        # 但为了保持类不变，我们尝试这种方式。

        # 确保数据集的根目录存在
        os.makedirs(suite_sparse_dataset_root, exist_ok=True)

        print(f"Initializing SuiteSparseDataset to fetch '{target_graph_info.name}'...")
        # 使用 target_graph_info.rows (节点数) 来限定搜索范围
        # 为了确保只获取这个图，我们可以将范围设得很窄
        node_count = target_graph_info.rows
        dataset_ss = SuiteSparseDataset(
            root=suite_sparse_dataset_root,
            min_nodes=node_count,
            max_nodes=node_count,
            limit=5  # 给一点余量，以防有其他同节点数的图排在前面
        )
        print(f"SuiteSparseDataset initialized. Found {len(dataset_ss.graph_list)} potential graphs in range.")

        # 在数据集中查找目标图
        # SuiteSparseDataset 在 process() 后会有一个 self.index 列表 (图的名称)
        if not dataset_ss.index:  # 如果索引为空，可能意味着数据尚未处理或找不到
            print("Warning: SuiteSparseDataset index is empty. Data might not be processed.")
            print(f"Please ensure '{target_graph_info.name}' was downloaded to '{dataset_ss.raw_dir}'")
            print(f"and processed into '{dataset_ss.processed_dir}'.")
            print(f"You might need to run the dataset script separately first if auto-download/process fails here.")

        found_in_dataset = False
        for i, name_in_index in enumerate(dataset_ss.index):
            if name_in_index == target_graph_info.name:
                graph_data_item_raw = dataset_ss[i]
                nx_graph_input = graph_data_item_raw.G  # 获取原始 NetworkX 图
                data_name_for_plot = graph_data_item_raw.name
                found_in_dataset = True
                print(f"Successfully loaded '{data_name_for_plot}' from processed SuiteSparseDataset.")
                break

        if not found_in_dataset:
            print(f"Error: Processed data for '{target_graph_info.name}' not found in SuiteSparseDataset index.")
            print(f"Available graphs in index: {dataset_ss.index if dataset_ss.index else 'None'}")
            print(
                f"Ensure the graph was downloaded to {dataset_ss.raw_dir} and processed to {dataset_ss.processed_dir}")
            exit()

    except ImportError:
        print("Error: ssgetpy library is not installed. Please install it via 'pip install ssgetpy'.")
        exit()
    except Exception as e:
        print(f"Error loading SuiteSparse graph '{graph_name_to_draw}': {e}")
        import traceback

        traceback.print_exc()
        exit()
else:
    # 为其他 graph_source 类型（如 "custom" 或 "rome"）保留现有逻辑
    print(f"Graph source '{graph_source}' is not 'suitesparse'. Using placeholder or other logic.")
    # 默认使用一个自定义图进行演示
    nx_graph_input = nx.star_graph(20)
    data_name_for_plot = "star_20_custom_fallback"
    nx_graph_input.graph['name'] = data_name_for_plot
    nx_graph_input.graph['dataset_name'] = 'custom_example'

if nx_graph_input is None or graph_data_item_raw is None:
    print("Error: Graph data could not be loaded or prepared.")
    exit()

print(
    f"Preparing data for graph: {data_name_for_plot} (Nodes: {nx_graph_input.number_of_nodes()}, Edges: {nx_graph_input.number_of_edges()})")

try:
    # graph_data_item_raw 已经是处理过的 GraphDrawingData 对象
    # 我们需要确保它在正确的设备上，并且有初始的 .pos (如果模型需要)
    # 对于从 SuiteSparseDataset 加载的图，它应该已经过了 pre_transform, static_transform, dynamic_transform
    graph_data_item = graph_data_item_raw.clone()  # 使用克隆以防修改原始缓存对象

    if graph_data_item.pos is None:
        print(f"Graph {data_name_for_plot} has no initial 'pos'. Generating random layout.")
        graph_data_item.pos = torch.rand((graph_data_item.num_nodes, 2)).float() * 10
    graph_data_item.pos = graph_data_item.pos.to(device)

    # 确保 edge_index (用于绘图)
    if not hasattr(graph_data_item, 'edge_index') or graph_data_item.edge_index is None:
        if hasattr(graph_data_item, 'perm_index') and hasattr(graph_data_item,
                                                              'edge_metaindex') and graph_data_item.edge_metaindex is not None:
            graph_data_item.edge_index = graph_data_item.perm_index[:, graph_data_item.edge_metaindex]
        else:
            undir_G_plot = nx_graph_input.to_undirected()
            if not undir_G_plot.edges():
                graph_data_item.edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_list_plot = list(undir_G_plot.edges())
                source_nodes_plot = [u for u, v in edge_list_plot];
                target_nodes_plot = [v for u, v in edge_list_plot]
                edge_index_numpy_plot = np.array(
                    [source_nodes_plot + target_nodes_plot, target_nodes_plot + source_nodes_plot])
                graph_data_item.edge_index = torch.from_numpy(edge_index_numpy_plot).long()
    graph_data_item.edge_index = graph_data_item.edge_index.to(device)

    # 准备 edge_attr for model (基于 perm_index 和 apsp_attr)
    if hasattr(graph_data_item, 'apsp_attr') and graph_data_item.apsp_attr is not None and \
            hasattr(graph_data_item, 'perm_index') and graph_data_item.perm_index is not None:
        apsp = graph_data_item.apsp_attr.float().to(device)
        if apsp.shape[0] == graph_data_item.perm_index.shape[1]:
            apsp_expanded = apsp[:, None]
            edge_attr = torch.cat([apsp_expanded, 1 / (apsp_expanded.square() + 1e-6)], dim=-1)
        else:
            print(f"Warning (SS): Mismatch apsp_attr length. Using dummy edge_attr.")
            edge_attr = torch.ones((graph_data_item.perm_index.shape[1], generator_params.edge_attr_dim),
                                   device=device).float()
    else:
        print("Warning (SS): apsp_attr or perm_index not found. Using dummy edge_attr.")
        # 确保 perm_index 存在，即使是虚拟的（如果 GraphDrawingData 没有正确填充）
        if not hasattr(graph_data_item, 'perm_index') or graph_data_item.perm_index is None:
            print("Error: perm_index is missing from graph_data_item. Cannot proceed with model.")
            exit()
        num_perms = graph_data_item.perm_index.shape[1]
        if not hasattr(graph_data_item, 'apsp_attr') or graph_data_item.apsp_attr is None:
            graph_data_item.apsp_attr = torch.ones(num_perms, device=device).float()  # Dummy apsp
        apsp_expanded = graph_data_item.apsp_attr.to(device).float()[:, None]
        edge_attr = torch.ones((num_perms, generator_params.edge_attr_dim), device=device).float()
        if generator_params.edge_attr_dim == 2:
            edge_attr = torch.cat([apsp_expanded, 1 / (apsp_expanded.square() + 1e-6)], dim=-1)
    edge_attr = edge_attr.to(device)

    data_list_for_batch = [graph_data_item.to(device)]
    batch = Batch.from_data_list(data_list_for_batch)

except Exception as e:
    print(f"Error preparing graph data for {data_name_for_plot}: {e}")
    import traceback

    traceback.print_exc()
    exit()

# --- 3. 使用模型生成布局 ---
print(f"Generating layout for {data_name_for_plot} using the pre-trained model...")
with torch.no_grad():
    canonicalizer_for_model_io = Compose(Center(), NormalizeRotation(), RescaleByStress())
    initial_positions_for_model = batch.pos.clone().to(device)  # 使用批处理中的 pos

    # 确保 canonicalizer 的所有输入都在正确的设备上
    batch_apsp_attr = batch.apsp_attr.to(device)
    batch_perm_index = batch.perm_index.to(device)
    batch_batch_index = batch.batch.to(device)

    init_pos_processed = canonicalizer_for_model_io(
        pos=initial_positions_for_model,
        apsp=batch_apsp_attr,
        edge_index=batch_perm_index,
        batch_index=batch_batch_index
    )
    predicted_layout_raw = generator(
        init_pos=init_pos_processed,
        edge_index=batch.perm_index,  # 模型使用 perm_index
        edge_attr=edge_attr,
        batch_index=batch.batch
    )
    print("Raw layout generated.")

# --- 4. 后处理布局 ---
final_layout = canonicalizer_for_model_io(
    pos=predicted_layout_raw,
    apsp=batch_apsp_attr,
    edge_index=batch_perm_index,
    batch_index=batch_batch_index
)
graph_data_item.pos = final_layout.cpu()  # 更新回CPU
print("Layout processed and finalized.")

# --- 5. 社区检测和节点颜色 ---
print("Performing community detection...")
node_colors_final = 'skyblue'
try:
    # 使用 nx_graph_input (加载时的原始 NetworkX 图)进行社区检测
    # 对于大图，Girvan-Newman 可能非常慢
    num_nodes_for_community = nx_graph_input.number_of_nodes()
    if num_nodes_for_community < 200:  # 阈值，可调整
        print(f"Running Girvan-Newman for {num_nodes_for_community} nodes...")
        # 确保是无向图，并且没有孤立节点（Girvan-Newman 可能对孤立节点行为不确定）
        # 如果图不是连通的，Girvan-Newman 会在每个连通分量上运行
        G_community = nx_graph_input.to_undirected()
        if not nx.is_connected(G_community) and num_nodes_for_community > 0:
            print("Warning: Graph for community detection is not connected. Results might be per component.")

        if num_nodes_for_community > 0:  # 确保图不为空
            comp_iter = girvan_newman(G_community)
            # 获取顶层社区划分（或者选择更细的划分）
            top_level_communities = tuple(sorted(c) for c in next(comp_iter))

            community_map = {}
            for i, comm in enumerate(top_level_communities):
                for node in comm:
                    community_map[node] = i  # 节点标签应与 nx_graph_input 中的一致

            unique_communities = sorted(list(set(community_map.values())))
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_for_communities = prop_cycle.by_key()['color']

            # 确保原始 nx_graph_input 的节点顺序与 graph_data_item.pos 的顺序一致
            # GraphDrawingData 在处理时可能会重新标记节点为 0..N-1
            # 因此，我们需要基于原始 nx_graph_input 的节点进行映射
            # 如果 nx_graph_input 的节点已经是 0..N-1，则可以直接使用

            # 获取 graph_data_item 使用的节点顺序（通常是 0 to N-1）
            # 我们假设 nx_graph_input 的节点已经是 GraphDrawingData 使用的节点标签
            # 如果不是，需要一个从原始图节点到 graph_data_item 节点索引的映射
            node_list_original = list(nx_graph_input.nodes())  # 获取原始节点顺序/标签

            node_colors_list = []
            for node_label in node_list_original:  # 遍历原始图的节点
                community_id = community_map.get(node_label, -1)  # 获取社区 ID
                color_index = community_id % len(colors_for_communities)
                node_colors_list.append(colors_for_communities[color_index])
            node_colors_final = node_colors_list

            print(f"Found {len(unique_communities)} communities.")
        else:
            print("Graph has no nodes, skipping community detection.")

    else:
        print(
            f"Graph with {num_nodes_for_community} nodes is large, skipping Girvan-Newman for speed. Using default node color.")
except Exception as e_comm:
    print(f"Error during community detection: {e_comm}. Using default node color.")
    import traceback

    traceback.print_exc()

# --- 6. 绘制图形 ---
print(f"Drawing the graph '{data_name_for_plot}'...")
plt.figure(figsize=(12, 10))  # 稍微大一点的图以便查看大图
title = f"Graph: {data_name_for_plot} ({graph_data_item.num_nodes}N, {nx_graph_input.number_of_edges()}E)\nLayout: SmartGD ({os.path.basename(generator_checkpoint_path)})"
plt.title(title)

node_size_dynamic = max(5, 1000 // graph_data_item.num_nodes if graph_data_item.num_nodes > 0 else 50)
edge_width_dynamic = max(0.2, 1.5 - graph_data_item.num_nodes / 200) if graph_data_item.num_nodes > 0 else 1.0

# graph_data_item.G 应该是 GraphDrawingData 内部转换和重新标记后的 NetworkX 图
# 为了确保社区颜色与绘制的节点对应，我们使用 graph_data_item.G 进行绘制
# 但社区检测是在 nx_graph_input 上做的。如果节点标签在 GraphDrawingData 中被重置为 0..N-1，
# 那么社区着色也应该基于这个 0..N-1 的顺序。
# 幸运的是，graph_data_item.draw() 期望的 pos 张量的顺序是 0..N-1。

graph_data_item.draw(attr={
    'node_color': node_colors_final,  # 使用基于社区的颜色
    'edge_color': 'grey',
    'node_size': node_size_dynamic,
    'with_labels': False,
    'width': edge_width_dynamic,
    'alpha': 0.7
})

plt.axis('equal')
# plt.axis('off') # 可以取消注释以获得更干净的图像
plt.tight_layout()
output_filename = f"{data_name_for_plot.replace('/', '_')}_{os.path.basename(generator_checkpoint_path).replace('.pt', '')}.png"
plt.savefig(output_filename)
print(f"Graph saved as {output_filename}")
plt.show()

# print(f"\nFinal node positions for '{data_name_for_plot}':")
# print(graph_data_item.pos.numpy())
# print(dgfs)