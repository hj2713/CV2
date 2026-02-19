import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# Plot the histogram of NNDR
def plot_histogram(THRESHOLD, ratios):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ratios, bins=50, edgecolor='black')
    ax.axvline(THRESHOLD, color='r', linestyle='--', label=f'threshold = {THRESHOLD}')
    ax.set_xlabel('NNDR (1NN / 2NN)')
    ax.set_ylabel('Count')
    ax.set_title('NNDR Histogram (L2 distance)')
    ax.legend()
    plt.savefig('results/step4_nndr_histogram.png', dpi=150)
    plt.show()

# Plot the connected descriptions
def plot_connected_descriptions(THRESHOLD, ratios, im1, im2, pts1, pts2, nn1_idx, nn2_idx):
    match_mask = ratios < THRESHOLD
    match_idx = np.where(match_mask)[0]
    unmatch_idx = np.where(~match_mask)[0]
    print(f'Threshold: {THRESHOLD} | Matched: {len(match_idx)} | Unmatched: {len(unmatch_idx)} | Total: {len(ratios)}')

    combined = np.concatenate([im1, im2], axis=1)
    offset = im1.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.imshow(combined)

    # Draw green lines for matches
    for i in match_idx:
        y1, x1 = pts1[:, i]
        y2, x2 = pts2[:, nn1_idx[i]]
        ax.plot([x1, x2 + offset], [y1, y2], 'g-', lw=0.8, alpha=0.6)

    # Matched points (green), unmatched (red)
    for i in match_idx:
        y1, x1 = pts1[:, i]
        y2, x2 = pts2[:, nn1_idx[i]]
        ax.plot(x1, y1, 'go', ms=4, mec='white', mew=0.5)
        ax.plot(x2 + offset, y2, 'go', ms=4, mec='white', mew=0.5)

    for i in unmatch_idx:
        y1, x1 = pts1[:, i]
        ax.plot(x1, y1, 'ro', ms=3, mec='white', mew=0.5)
    ax.set_title(f'Feature Matching — Matched: {len(match_idx)} | Unmatched: {len(unmatch_idx)} | Total: {len(ratios)}')
    ax.axis('off')
    plt.savefig('results/step4_matches.png', dpi=150)
    plt.show()

# Extract RGB patch
def extract_rgb_patch(img, y, x, patch_size=40, out_size=8):
    """Extract a patch_size x patch_size RGB window, resize to out_size x out_size."""
    r = patch_size // 2
    h, w = img.shape[:2]
    y, x = int(y), int(x)
    y0 = max(y - r, 0)
    y1 = min(y + r, h)
    x0 = max(x - r, 0)
    x1 = min(x + r, w)
    window = img[y0:y1, x0:x1]
    return resize(window, (out_size, out_size, 3), anti_aliasing=True)

# Plot the top feature matches
def plot_top_feature_matches(ratios, match_mask, pts1, pts2, nn1_idx, nn2_idx, match_idx, im1, im2):
    top_order = np.argsort(ratios[match_mask])
    top_k = min(5, len(top_order))

    fig, axes = plt.subplots(top_k, 3, figsize=(9, 3 * top_k))
    if top_k == 1:
        axes = axes.reshape(1, 3)

    col_headers = ['Image 1 Feature', 'Nearest Neighbor (1st)', 'Second Nearest (2nd)']

    for rank in range(top_k):
        idx1 = match_idx[top_order[rank]]
        i2_1nn = nn1_idx[idx1]
        i2_2nn = nn2_idx[idx1]
        nndr_val = ratios[idx1]

        # Get RGB patches from original color images
        y1, x1 = pts1[:, idx1]
        y2_1, x2_1 = pts2[:, i2_1nn]
        y2_2, x2_2 = pts2[:, i2_2nn]

        patch_img1 = extract_rgb_patch(im1, y1, x1)
        patch_1nn  = extract_rgb_patch(im2, y2_1, x2_1)
        patch_2nn  = extract_rgb_patch(im2, y2_2, x2_2)

        patches = [patch_img1, patch_1nn, patch_2nn]
        sub_titles = [f'Image 1 Feature #{rank+1}', f'Image 2 NN (1st)', f'Image 2 2NN (2nd)']

        for col in range(3):
            axes[rank, col].imshow(patches[col])
            axes[rank, col].set_title(sub_titles[col], fontsize=9)
            axes[rank, col].axis('off')

        # NNDR label at top-left of the first column
        axes[rank, 0].text(
            0.02, 0.95, f'NNDR={nndr_val:.3f}',
            transform=axes[rank, 0].transAxes,
            fontsize=9, fontweight='bold', color='white',
            va='top', ha='left',

            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8)
        )

    # Column super-headers
    fig.text(0.22, 0.98, col_headers[0], ha='center', fontsize=11, fontweight='bold')
    fig.text(0.52, 0.98, col_headers[1], ha='center', fontsize=11, fontweight='bold')
    fig.text(0.82, 0.98, col_headers[2], ha='center', fontsize=11, fontweight='bold')

    fig.suptitle('Top 5 Best Feature Matches by NNDR (L2) - RGB', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('results/step4_top5_matches.png', dpi=150, bbox_inches='tight')
    plt.show()