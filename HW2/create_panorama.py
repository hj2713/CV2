from skimage.transform import ProjectiveTransform, warp
from skimage.measure import ransac

# --- Load 3 images ---
imC1 = load_image('images/imgC1.jpg')
imC2 = load_image('images/imgC2.jpg')
imC3 = load_image('images/imgC3.jpg')
gC1, gC2, gC3 = color.rgb2gray(imC1[:,:,:3]), color.rgb2gray(imC2[:,:,:3]), color.rgb2gray(imC3[:,:,:3])

# --- Run pipeline on each image: Harris → NMS → Descriptors ---
h_1, c_1 = harris.get_harris_corners(gC1, edge_discard=20)
h_2, c_2 = harris.get_harris_corners(gC2, edge_discard=20)
h_3, c_3 = harris.get_harris_corners(gC3, edge_discard=20)

nC1, nC2_a = apply_nms(h_1, c_1, h_2, c_2)
_, nC3 = apply_nms(h_2, c_2, h_3, c_3)
nC2_b = nC2_a  # same NMS result for imgC2

dC1, pC1 = extract_descriptors(imC1, nC1)
dC2, pC2 = extract_descriptors(imC2, nC2_a)
dC3, pC3 = extract_descriptors(imC3, nC3)

# --- Match pairs: (C1,C2) and (C2,C3) ---
def match_pair(descA, descB):
    from scipy.spatial.distance import cdist
    dists = cdist(descA, descB, metric='euclidean')
    ratios, nn1, nn2 = [], [], []
    for i in range(dists.shape[0]):
        idx = np.argsort(dists[i])
        ratios.append(dists[i, idx[0]] / (dists[i, idx[1]] + 1e-8))
        nn1.append(idx[0]); nn2.append(idx[1])
    return np.array(ratios), np.array(nn1), np.array(nn2)

T = 0.7
r12, nn1_12, nn2_12 = match_pair(dC1, dC2)
r23, nn1_23, nn2_23 = match_pair(dC2, dC3)
m12 = np.where(r12 < T)[0]
m23 = np.where(r23 < T)[0]
print(f'C1↔C2: {len(m12)} matches | C2↔C3: {len(m23)} matches')

# --- RANSAC homographies ---
def get_homography(ptsA, ptsB, nn1, matches):
    src = np.array([[ptsB[1, nn1[i]], ptsB[0, nn1[i]]] for i in matches])
    dst = np.array([[ptsA[1, i], ptsA[0, i]] for i in matches])
    model, inliers = ransac((src, dst), ProjectiveTransform,
                            min_samples=4, residual_threshold=4, max_trials=2000)
    print(f'  RANSAC inliers: {inliers.sum()}/{len(inliers)}')
    return model, inliers

print('H12 (C2→C1):')
H12, inliers12 = get_homography(pC1, pC2, nn1_12, m12)
print('H23 (C3→C2):')
H23, inliers23 = get_homography(pC2, pC3, nn1_23, m23)

# Chain: C3→C1 = H12 @ H23
H13 = ProjectiveTransform(matrix=H12.params @ H23.params)

# --- Compute canvas and warp ---
h1, w1 = imC1.shape[:2]
h2, w2 = imC2.shape[:2]
h3, w3 = imC3.shape[:2]

corners = np.vstack([
    [[0,0],[w1,0],[w1,h1],[0,h1]],
    H12(np.array([[0,0],[w2,0],[w2,h2],[0,h2]])),
    H13(np.array([[0,0],[w3,0],[w3,h3],[0,h3]]))
])
mn = np.floor(corners.min(axis=0)).astype(int)
mx = np.ceil(corners.max(axis=0)).astype(int)
out_w, out_h = mx[0]-mn[0], mx[1]-mn[1]

shift = np.array([[1,0,-mn[0]],[0,1,-mn[1]],[0,0,1]], dtype=float)

wC2 = warp(imC2, ProjectiveTransform(matrix=shift @ H12.params).inverse,
           output_shape=(out_h, out_w), preserve_range=True).astype(np.uint8)
wC3 = warp(imC3, ProjectiveTransform(matrix=shift @ H13.params).inverse,
           output_shape=(out_h, out_w), preserve_range=True).astype(np.uint8)

# Place C1 and blend
canvas = np.zeros((out_h, out_w, 3), dtype=np.float64)
counts = np.zeros((out_h, out_w), dtype=np.float64)

yo, xo = -mn[1], -mn[0]
canvas[yo:yo+h1, xo:xo+w1] += imC1[:,:,:3].astype(float)
counts[yo:yo+h1, xo:xo+w1] += 1

for w_img in [wC2, wC3]:
    mask = w_img.sum(axis=2) > 0
    canvas[mask] += w_img[mask].astype(float)
    counts[mask] += 1

counts[counts == 0] = 1
result = (canvas / counts[:,:,None]).astype(np.uint8)

# --- Display: Row 1 = 3 images, Row 2 = panorama ---
fig, axes = plt.subplots(2, 1, figsize=(18, 12),
                         gridspec_kw={'height_ratios': [1, 1.5]})
# Row 1: originals side by side
row1 = np.concatenate([imC1[:,:,:3], imC2[:,:,:3], imC3[:,:,:3]], axis=1)
axes[0].imshow(row1)
axes[0].set_title('Original Images: imgC1 | imgC2 | imgC3')
axes[0].axis('off')
# Row 2: panorama
axes[1].imshow(result)
axes[1].set_title('Stitched Panorama')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('results/extra_credit_panorama.png', dpi=150, bbox_inches='tight')
plt.show()
