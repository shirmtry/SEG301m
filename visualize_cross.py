# -*- coding: utf-8 -*-
"""visualize_cross.py — Vẽ biểu đồ network các cluster từ cross_analysis.json"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


TOPIC_COLORS = {
    'Lao động':              '#1976D2',
    'Hôn nhân & Gia đình':   '#E91E63',
    'Giao thông':            '#FF9800',
    'Hình sự':               '#D32F2F',
    'Dân sự':                '#388E3C',
    'Đất đai':               '#795548',
    'Doanh nghiệp & Thuế':   '#7B1FA2',
    'Hành chính':            '#0097A7',
    'Khác':                  '#9E9E9E',
}


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_graph(data):
    """Xây đồ thị từ overlap_pairs."""
    G = nx.Graph()
    # Thêm node từ clusters
    for cl in data.get('clusters', []):
        for qid in cl['ids']:
            G.add_node(qid)

    # Thêm edge với weight = jaccard
    for p in data.get('overlap_pairs', []):
        G.add_edge(p['id_a'], p['id_b'],
                   weight=p['jaccard'],
                   cross=p['cross_topic'],
                   topic_a=p['topic_a'],
                   topic_b=p['topic_b'])

    return G


def get_topic(qid, records_map):
    return records_map.get(qid, {}).get('hint_topic', 'Khác')


def draw_network(G, records_map, out_path):
    """Vẽ network đồ thị, node = câu hỏi, màu theo topic."""
    if len(G.nodes) == 0:
        print('⚠️ Không có node — bỏ qua vẽ network.')
        return

    fig, ax = plt.subplots(figsize=(18, 12))
    pos = nx.spring_layout(G, k=1.2, iterations=80, seed=42)

    # Node colors
    node_colors = [TOPIC_COLORS.get(get_topic(n, records_map), '#9E9E9E')
                   for n in G.nodes]

    # Edge weights → width
    weights = [G[u][v]['weight'] for u, v in G.edges]
    widths = [w * 6 for w in weights]

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                           alpha=0.35, edge_color='#555555')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900,
                           node_color=node_colors,
                           edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                            font_weight='bold', font_color='white')

    # Legend
    handles = [mpatches.Patch(color=c, label=t)
               for t, c in TOPIC_COLORS.items()
               if t in set(get_topic(n, records_map) for n in G.nodes)]
    ax.legend(handles=handles, loc='upper left', fontsize=10,
              framealpha=0.9)

    ax.set_title('Mạng lưới câu hỏi trùng nội dung\n'
                 f'({len(G.nodes)} node, {len(G.edges)} cạnh)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f'✅ Đã lưu: {out_path}')


def draw_cluster_bars(data, out_path):
    """Vẽ bar chart các cluster theo kích thước."""
    clusters = data.get('clusters', [])
    if not clusters:
        print('⚠️ Không có cluster.')
        return

    # Sắp xếp theo kích thước giảm dần
    clusters = sorted(clusters, key=lambda c: len(c['ids']), reverse=True)

    labels = [f"#{c['ids'][0]}-{c['ids'][-1]} ({len(c['ids'])} câu)"
              for c in clusters]
    sizes = [len(c['ids']) for c in clusters]

    fig, ax = plt.subplots(figsize=(14, max(6, len(clusters) * 0.45)))
    colors_bar = plt.cm.viridis(
        [s / max(sizes) for s in sizes]) if sizes else []

    bars = ax.barh(labels, sizes, color=colors_bar)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(size), va='center', fontweight='bold')

    ax.set_xlabel('Số câu trong cluster', fontsize=12)
    ax.set_title(f'Phân bố {len(clusters)} cluster câu hỏi trùng nội dung',
                 fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✅ Đã lưu: {out_path}')


def draw_cross_topic_matrix(data, out_path):
    """Vẽ ma trận câu hỏi cross-topic."""
    cross = data.get('cross_topic', [])
    if not cross:
        print('⚠️ Không có cross-topic.')
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(cross) * 0.7)))

    questions = [f"#{r['id']}: {r['question'][:50]}..." for r in cross]
    primary = [r['primary_topic'] for r in cross]
    secondary = [' + '.join(r['secondary_topics']) for r in cross]

    y_pos = range(len(cross))
    colors_p = [TOPIC_COLORS.get(t, '#9E9E9E') for t in primary]

    ax.barh(y_pos, [1] * len(cross), color=colors_p, alpha=0.8,
            label='Chủ đề chính')
    ax.barh(y_pos, [0.5] * len(cross), left=[1.05] * len(cross),
            color='#FFC107', alpha=0.7, label='Chủ đề phụ')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(questions, fontsize=9)
    ax.set_xlim(0, 1.7)
    ax.set_xticks([])
    ax.set_title('Câu hỏi cross-topic (đa chủ đề)',
                 fontsize=14, fontweight='bold', pad=15)

    # Text chú thích
    for i, (p, s) in enumerate(zip(primary, secondary)):
        ax.text(0.5, i, p, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(1.3, i, s, ha='center', va='center',
                fontsize=9, color='#333333')

    ax.invert_yaxis()
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✅ Đã lưu: {out_path}')


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, 'cross_analysis.json')
    out_dir = os.path.dirname(json_path)

    data = load_data(json_path)

    # Lấy thông tin topic cho từng ID từ cross_topic
    records_map = {r['id']: {'hint_topic': r['hint_topic']}
                   for r in data.get('cross_topic', [])}
    # Bổ sung từ clusters (fallback 'Khác')
    for cl in data.get('clusters', []):
        for qid in cl['ids']:
            records_map.setdefault(qid, {'hint_topic': 'Khác'})

    print('🎨 Đang vẽ biểu đồ...')
    draw_network(build_graph(data), records_map,
                 os.path.join(out_dir, 'cross_network.png'))
    draw_cluster_bars(data, os.path.join(out_dir, 'cross_clusters.png'))
    draw_cross_topic_matrix(data, os.path.join(out_dir, 'cross_topics.png'))
    print('🎉 Hoàn tất!')


if __name__ == '__main__':
    main()