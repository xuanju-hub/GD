import torch
from torch_geometric.data import Batch # <--- 修正导入
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import os # 用于检查文件路径

# 从 SmartGD 框架导入必要的类
from smartgd.model import Generator
from smartgd.data import GraphDrawingData
from smartgd.datasets import RomeDataset # 可选，用于加载标准数据集
from smartgd.transformations import Compose, Center, NormalizeRotation, RescaleByStress

# --- 0. 配置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# <<<--- 修改这里：设置您的预训练模型检查点路径 ---<<<
generator_checkpoint_path = "./generator_stress_only.pt" # 默认

# 模型参数
generator_params = Generator.Params(
    num_blocks=11,
    block_depth=3,
    block_width=8,
    block_output_dim=8,
    edge_net_depth=2,
    edge_net_width=16,
    edge_attr_dim=2,
    node_attr_dim=2,
)

# --- 1. 加载模型 ---
print(f"Loading generator model from: {generator_checkpoint_path}")
if not os.path.exists(generator_checkpoint_path):
    print(f"Error: Generator checkpoint not found at {generator_checkpoint_path}.")
    exit()

generator = Generator(params=generator_params).to(device)
try:
    # 使用 weights_only=True
    generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=device, weights_only=True))
    generator.eval()
    print("Generator model loaded successfully.")
except Exception as e:
    print(f"Error loading generator model: {e}")
    exit()

# --- 2. 准备数据 ---
nx_graph_example = nx.path_graph(7)
nx_graph_example.graph['name'] = 'path_7_example'
nx_graph_example.graph['dataset_name'] = 'synthetic_example'

GraphDrawingData.set_optional_fields([
    "edge_pair_metaindex",
])

try:
    graph_data_item = GraphDrawingData(G=nx_graph_example)
    graph_data_item = graph_data_item.pre_transform().static_transform().dynamic_transform()

    if graph_data_item.pos is None:
        graph_data_item.pos = torch.rand((graph_data_item.num_nodes, 2)).float() * 10
    graph_data_item.pos = graph_data_item.pos.to(device)

    if not hasattr(graph_data_item, 'edge_index') or graph_data_item.edge_index is None:
        if hasattr(graph_data_item, 'perm_index') and hasattr(graph_data_item, 'edge_metaindex') and graph_data_item.edge_metaindex is not None:
            graph_data_item.edge_index = graph_data_item.perm_index[:, graph_data_item.edge_metaindex]
        else:
            undir_G = nx_graph_example.to_undirected() # 确保是无向图以获取所有边
            if not undir_G.edges():
                 graph_data_item.edge_index = torch.empty((2,0), dtype=torch.long)
            else:
                edge_list = list(undir_G.edges())
                source_nodes = [u for u, v in edge_list]
                target_nodes = [v for u, v in edge_list]
                # PyG 通常期望无向图的边是双向列出的
                edge_index_numpy = np.array([source_nodes + target_nodes, target_nodes + source_nodes])
                graph_data_item.edge_index = torch.from_numpy(edge_index_numpy).long()
    graph_data_item.edge_index = graph_data_item.edge_index.to(device)


    if hasattr(graph_data_item, 'apsp_attr') and graph_data_item.apsp_attr is not None:
        apsp = graph_data_item.apsp_attr.float().to(device)
        if apsp.shape[0] == graph_data_item.perm_index.shape[1]:
            apsp_expanded = apsp[:, None]
            edge_attr = torch.cat([apsp_expanded, 1 / (apsp_expanded.square() + 1e-6)], dim=-1)
        else:
            print(f"Warning: Mismatch in apsp_attr length ({apsp.shape[0]}) and perm_index count ({graph_data_item.perm_index.shape[1]}). Using dummy edge_attr.")
            num_permutations = graph_data_item.perm_index.shape[1]
            edge_attr = torch.ones((num_permutations, generator_params.edge_attr_dim), device=device).float()
    else:
        print("Warning: apsp_attr not found. Generating dummy apsp_attr and edge_attr.")
        num_permutations = graph_data_item.perm_index.shape[1]
        # 确保 apsp_attr 被正确创建，即使是虚拟的
        graph_data_item.apsp_attr = torch.ones(num_permutations, device=device).float()
        apsp_expanded = graph_data_item.apsp_attr[:, None] # 使用新创建的 apsp_attr
        edge_attr = torch.ones((num_permutations, generator_params.edge_attr_dim), device=device).float()
        if generator_params.edge_attr_dim == 2:
             edge_attr = torch.cat([apsp_expanded, 1 / (apsp_expanded.square() + 1e-6)], dim=-1)
    edge_attr = edge_attr.to(device)


    data_list_for_batch = [graph_data_item.to(device)]
    batch = Batch.from_data_list(data_list_for_batch) # <--- 使用修正后的方法

except Exception as e:
    print(f"Error preparing graph data: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 3. 使用模型生成布局 ---
print("Generating layout using the pre-trained model...")
with torch.no_grad():
    canonicalizer_for_model_io = Compose(
        Center(),
        NormalizeRotation(),
        RescaleByStress(),
    )

    initial_positions_for_model = batch.pos.clone().to(device)

    # 确保 canonicalizer_for_model_io 的所有输入张量都在正确的设备上
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
        edge_index=batch.perm_index.to(device), # 确保在设备上
        edge_attr=edge_attr.to(device),         # 确保在设备上
        batch_index=batch.batch.to(device)      # 确保在设备上
    )
    print("Raw layout generated by model.")

# --- 4. 后处理生成的布局 ---
final_layout = canonicalizer_for_model_io(
    pos=predicted_layout_raw,
    apsp=batch_apsp_attr,
    edge_index=batch_perm_index,
    batch_index=batch_batch_index
)
graph_data_item.pos = final_layout.cpu()
print("Layout processed and finalized.")

# # --- 5. 绘制图形 ---
# print(f"Drawing the graph '{graph_data_item.name}' with the generated layout...")
# plt.figure(figsize=(10, 8))
# plt.title(f"Graph: {graph_data_item.name} - Layout by SmartGD ({os.path.basename(generator_checkpoint_path)})")
#
# graph_data_item.draw(attr={'node_color': 'skyblue',
#                            'edge_color': 'black',
#                            'node_size': 200,
#                            'with_labels': True,
#                            'font_size': 10,
#                            'width': 1.5})
#
# plt.xlabel("X-coordinate")
# plt.ylabel("Y-coordinate")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.axis('equal')
# plt.show()
#
# print(f"\nFinal node positions for '{graph_data_item.name}':")
# print(graph_data_item.pos.numpy())

# (可选的 RomeDataset 示例代码可以放在这里，同样需要应用 Batch.from_data_list 修复)
# --- 5. 绘制图形 ---
# --- 示例：如何使用 RomeDataset 加载和绘制 ---
rome_dataset_root = "./datasets" # SmartGD 中定义的 DATASET_ROOT
# smartgd_demo.ipynb 从 "assets/rome_index.txt" 加载索引
demo_rome_index_path = "assets/rome_index.txt"

if not os.path.exists(demo_rome_index_path):
    print(f"Warning: Demo Rome index '{demo_rome_index_path}' not found. Skipping RomeDataset example.")
else:
    print("\n--- Drawing graphs from RomeDataset ---")
    try:
        rome_index_df = pd.read_csv(demo_rome_index_path, header=None)
        # 实例化 RomeDataset
        # 您可以根据需要筛选 rome_index_df[0] 来选择特定的图
        # 例如，选择索引中的前5个图，或者根据图的名称筛选
        selected_graph_indices = rome_index_df[0][:5] # <--- 修改这里来选择不同的图或更多图

        print(f"Loading selected RomeDataset graphs (this may take time if it's the first run)...")
        dataset_rome = RomeDataset(root=rome_dataset_root, index=selected_graph_indices)

        if len(dataset_rome) > 0:
            for i, data_item_rome in enumerate(dataset_rome):
                print(f"\nProcessing Rome graph: {data_item_rome.name} ({data_item_rome.num_nodes} nodes)")
                data_item_rome = data_item_rome.to(device)

                # 准备模型输入 (类似于主示例中的单个图)
                current_batch_list_rome = [data_item_rome]
                current_batch_rome = Batch.from_data_list(current_batch_list_rome)

                if current_batch_rome.pos is None:
                    # RomeDataset 中的图通常在 static_transform 阶段会有一个 pos (可能是随机的)
                    # 如果没有，则需要生成一个
                    print(f"Graph {data_item_rome.name} has no initial 'pos'. Generating random layout.")
                    current_batch_rome.pos = torch.rand((current_batch_rome.num_nodes, 2), device=device).float() * 10
                else:
                    # 确保 pos 在正确的设备上
                    current_batch_rome.pos = current_batch_rome.pos.to(device)


                # 准备 edge_attr (确保所有必需的属性都存在于 data_item_rome 中)
                if hasattr(current_batch_rome, 'apsp_attr') and current_batch_rome.apsp_attr is not None:
                    apsp_b_rome = current_batch_rome.apsp_attr.float().to(device)[:, None]
                    if apsp_b_rome.shape[0] == current_batch_rome.perm_index.shape[1]:
                        edge_attr_b_rome = torch.cat([apsp_b_rome, 1 / (apsp_b_rome.square() + 1e-6)], dim=-1)
                    else:
                        print(f"Warning (Rome): Mismatch in apsp_attr length ({apsp_b_rome.shape[0]}) and perm_index count ({current_batch_rome.perm_index.shape[1]}). Using dummy edge_attr.")
                        edge_attr_b_rome = torch.ones((current_batch_rome.perm_index.shape[1], generator_params.edge_attr_dim), device=device).float()
                else:
                    print(f"Warning (Rome): apsp_attr not found for {data_item_rome.name}. Using dummy edge_attr.")
                    # 如果 apsp_attr 不存在，我们需要创建一个
                    num_permutations_rome = current_batch_rome.perm_index.shape[1]
                    current_batch_rome.apsp_attr = torch.ones(num_permutations_rome, device=device).float() # Dummy apsp
                    apsp_expanded_rome = current_batch_rome.apsp_attr[:, None]
                    edge_attr_b_rome = torch.ones((num_permutations_rome, generator_params.edge_attr_dim), device=device).float()
                    if generator_params.edge_attr_dim == 2:
                         edge_attr_b_rome = torch.cat([apsp_expanded_rome, 1 / (apsp_expanded_rome.square() + 1e-6)], dim=-1)
                edge_attr_b_rome = edge_attr_b_rome.to(device)


                init_pos_processed_b_rome = canonicalizer_for_model_io(
                    pos=current_batch_rome.pos.clone(), # 使用当前图的 pos
                    apsp=current_batch_rome.apsp_attr.to(device),
                    edge_index=current_batch_rome.perm_index.to(device),
                    batch_index=current_batch_rome.batch.to(device)
                )

                with torch.no_grad():
                    pred_layout_b_rome = generator(
                        init_pos=init_pos_processed_b_rome,
                        edge_index=current_batch_rome.perm_index.to(device),
                        edge_attr=edge_attr_b_rome,
                        batch_index=current_batch_rome.batch.to(device)
                    )

                final_layout_b_rome = canonicalizer_for_model_io(
                    pos=pred_layout_b_rome,
                    apsp=current_batch_rome.apsp_attr.to(device),
                    edge_index=current_batch_rome.perm_index.to(device),
                    batch_index=current_batch_rome.batch.to(device)
                )
                data_item_rome.pos = final_layout_b_rome.cpu() # 更新回 CPU

                plt.figure(figsize=(7, 7))
                plt.title(f"Rome Graph: {data_item_rome.name} ({data_item_rome.num_nodes} nodes) - SmartGD Layout")
                data_item_rome.draw(attr={'node_size': max(10, 1000 // data_item_rome.num_nodes if data_item_rome.num_nodes > 0 else 70),
                                          'with_labels': data_item_rome.num_nodes < 30, # 小图显示标签
                                          'font_size': 8})
                plt.show()
        else:
            print("Selected RomeDataset slice is empty or could not be loaded.")
    except Exception as e_rome:
        print(f"An error occurred while loading or drawing from RomeDataset: {e_rome}")
        import traceback
        traceback.print_exc()