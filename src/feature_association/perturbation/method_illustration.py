"""
Simple illustration of Fisher's exact test for rejuvenating vs accelerating drugs
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Color scheme
color_reversal = '#95E1D3'  # Light teal for reversal
color_accel = '#FFE66D'     # Yellow for acceleration

# ============================================================================
# TITLE
# ============================================================================
ax.text(5, 7.5, "Fisher's Exact Test: Rejuvenating vs Accelerating Drugs", 
        fontsize=16, weight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))

# ============================================================================
# CONTINGENCY TABLE
# ============================================================================
table_x = 2.5
table_y = 5

ax.text(5, 6.3, "2×2 Contingency Table for Fisher's Test", 
        fontsize=13, weight='bold', ha='center')

# Column headers
ax.text(table_x+0.9, table_y+0.6, "Aging: ↑ TFs", fontsize=11, ha='center', weight='bold', color='darkred')
ax.text(table_x+2.4, table_y+0.6, "Aging: ↓ TFs", fontsize=11, ha='center', weight='bold', color='darkblue')

# Row headers
ax.text(table_x-0.7, table_y, "Drug: ↑ TFs", fontsize=11, ha='right', weight='bold', color='darkred')
ax.text(table_x-0.7, table_y-1.0, "Drug: ↓ TFs", fontsize=11, ha='right', weight='bold', color='darkblue')

# Cell a (top-left): ACCELERATION - both increase
rect_a = Rectangle((table_x, table_y-0.3), 1.3, 0.6, 
                    facecolor=color_accel, edgecolor='black', linewidth=2)
ax.add_patch(rect_a)
ax.text(table_x+0.65, table_y+0.1, "a", fontsize=18, ha='center', weight='bold')
ax.text(table_x+0.65, table_y-0.15, "ACCELERATION", fontsize=9, ha='center', weight='bold')

# Cell c (top-right): REVERSAL - age decreases, drug increases
rect_c = Rectangle((table_x+1.5, table_y-0.3), 1.3, 0.6,
                    facecolor=color_reversal, edgecolor='green', linewidth=3)
ax.add_patch(rect_c)
ax.text(table_x+2.15, table_y+0.1, "c", fontsize=18, ha='center', weight='bold')
ax.text(table_x+2.15, table_y-0.15, "REVERSAL", fontsize=9, ha='center', 
        color='darkgreen', weight='bold')

# Cell b (bottom-left): REVERSAL - age increases, drug decreases
rect_b = Rectangle((table_x, table_y-1.3), 1.3, 0.6,
                    facecolor=color_reversal, edgecolor='green', linewidth=3)
ax.add_patch(rect_b)
ax.text(table_x+0.65, table_y-0.9, "b", fontsize=18, ha='center', weight='bold')
ax.text(table_x+0.65, table_y-1.15, "REVERSAL", fontsize=9, ha='center', 
        color='darkgreen', weight='bold')

# Cell d (bottom-right): ACCELERATION - both decrease
rect_d = Rectangle((table_x+1.5, table_y-1.3), 1.3, 0.6,
                    facecolor=color_accel, edgecolor='black', linewidth=2)
ax.add_patch(rect_d)
ax.text(table_x+2.15, table_y-0.9, "d", fontsize=18, ha='center', weight='bold')
ax.text(table_x+2.15, table_y-1.15, "ACCELERATION", fontsize=9, ha='center', weight='bold')

# ============================================================================
# INTERPRETATION
# ============================================================================
interpret_y = 3.2

ax.text(5, interpret_y, "Drug Classification", fontsize=13, weight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='black', linewidth=1.5))

# Reversal score formula
ax.text(5, interpret_y-0.6, "Reversal Score = ", fontsize=11, ha='center', weight='bold')
ax.text(5, interpret_y-1.0, "(b + c) - (a + d)", fontsize=14, ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))
ax.text(5, interpret_y-1.3, "(b + c) + (a + d)", fontsize=14, ha='center')

ax.text(5, interpret_y-1.8, "Range: -1 (complete acceleration) to +1 (complete reversal)", 
        fontsize=9, ha='center', style='italic')

# Classification criteria
criteria_y = interpret_y - 2.5

# Rejuvenating
ax.text(1.5, criteria_y+0.2, "✓ Rejuvenating:", fontsize=11, weight='bold', ha='left', color='darkgreen')
ax.text(1.8, criteria_y-0.1, "• Score > 0.2", fontsize=10, ha='left')
ax.text(1.8, criteria_y-0.4, "• p < 0.05 (Fisher's test)", fontsize=10, ha='left')
ax.text(1.8, criteria_y-0.7, "• ≥3 common TFs", fontsize=10, ha='left')
rect_rej = Rectangle((1.2, criteria_y-0.8), 0.2, 0.2, facecolor=color_reversal, 
                      edgecolor='green', linewidth=2)
ax.add_patch(rect_rej)

# Neutral
ax.text(5, criteria_y+0.2, "○ Neutral:", fontsize=11, weight='bold', ha='left', color='gray')
ax.text(5.3, criteria_y-0.1, "• |Score| ≤ 0.2", fontsize=10, ha='left')
ax.text(5.3, criteria_y-0.4, "• or p ≥ 0.05", fontsize=10, ha='left')

# Accelerating
ax.text(8, criteria_y+0.2, "✗ Accelerating:", fontsize=11, weight='bold', ha='left', color='darkred')
ax.text(8.3, criteria_y-0.1, "• Score < -0.2", fontsize=10, ha='left')
ax.text(8.3, criteria_y-0.4, "• p < 0.05 (Fisher's test)", fontsize=10, ha='left')
ax.text(8.3, criteria_y-0.7, "• ≥3 common TFs", fontsize=10, ha='left')
rect_acc = Rectangle((7.7, criteria_y-0.8), 0.2, 0.2, facecolor=color_accel, 
                      edgecolor='red', linewidth=2)
ax.add_patch(rect_acc)

plt.tight_layout()
plt.savefig('/Users/jno24/Documents/projs/ongoing/ciim/base_folder/output/plots/perturbations/fisher_test_illustration.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/jno24/Documents/projs/ongoing/ciim/base_folder/output/plots/perturbations/fisher_test_illustration.pdf',
            bbox_inches='tight', facecolor='white')
print("Fisher's test illustration saved to:")
print("  - fisher_test_illustration.png")
print("  - fisher_test_illustration.pdf")
plt.show()
