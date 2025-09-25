import torch
import plotly.graph_objects as go
import splat_mesh_helpers as splt

def plotly_visualize_splat(file_path):
    pos3D, _, colors, _, _, _, _, _ = splt.splat_unpacker_threshold_return_mindistance(25, file_path, 100)

    # 1. 过滤异常值（移除无效坐标）
    valid_mask = torch.isfinite(pos3D).all(dim=1) & torch.isfinite(colors).all(dim=1)
    pos3D = pos3D[valid_mask]
    colors = colors[valid_mask]

    # 2. 降采样（每10个点取1个，减少密度）
    sample_rate = 10  # 可调整（值越大点越少）
    pos3D = pos3D[::sample_rate]
    colors = colors[::sample_rate]

    # 3. 确保颜色是RGB格式（移除透明度通道）
    if colors.shape[1] >= 3:
        colors_rgb = torch.clamp(colors[:, :3], 0, 1)  # 仅保留RGB通道
    else:
        colors_rgb = torch.clamp(colors, 0, 1)  # 处理纯RGB数据

    # 创建 3D 散点图
    fig = go.Figure(data=[
        go.Scatter3d(
            x=pos3D[:, 0], y=pos3D[:, 1], z=pos3D[:, 2],
            mode='markers',
            marker=dict(
                size=3,  # 增大点大小
                color=colors_rgb,  # 使用修正后的RGB颜色
                opacity=0.6  # 降低透明度减少重叠干扰
            )
        )
    ])

    # 4. 优化视角和坐标轴
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)  # 调整初始视角
            )
        ),
        title=f"Splat Visualization (N={len(pos3D)})"
    )

    fig.show(renderer='browser')


# 调用示例
if __name__ == '__main__':
    try:
        plotly_visualize_splat("data/content_splats/horse.ply")
        # 阻塞进程，防止程序立即退出（等待用户手动关闭）
        input("可视化完成，按 Enter 键退出...")
    except Exception as e:
        # 捕获并显示异常，避免程序静默退出
        print(f"运行出错: {str(e)}")
        input("按 Enter 键退出...")