
import os
import glob
import urllib.parse
from datetime import datetime

# Find the output directory
base_dir = '/Users/himanshujhawar/Downloads/CV2/HW1'
# Use the known relative directory name
out_dir_name = 'output_images'
print(f"Using output directory: {out_dir_name}")

# Offsets from offset.txt
# Mapping basenames (without extension or with .tif/.jpg) to offsets
raw_offsets = {
    'church': {'G': '(25, 4)', 'R': '(58, -4)'},
    'emir': {'G': '(49, 24)', 'R': '(107, 40)'},
    'harvesters': {'G': '(60, 17)', 'R': '(124, 14)'},
    'icon': {'G': '(42, 17)', 'R': '(90, 23)'},
    'italil': {'G': '(38, 22)', 'R': '(77, 36)'},
    'lastochikino': {'G': '(-3, -2)', 'R': '(76, -8)'},
    'lugano': {'G': '(41, -17)', 'R': '(92, -29)'},
    'master-pnp-prok-00000-00082a': {'G': '(32, 4)', 'R': '(79, 7)'},
    'master-pnp-prok-00100-00172a': {'G': '(39, -1)', 'R': '(151, -7)'},
    'master-pnp-prok-00100-00187a': {'G': '(33, -11)', 'R': '(139, -26)'},
    'master-pnp-prok-00100-00189a': {'G': '(25, -18)', 'R': '(116, -38)'},
    'melons': {'G': '(80, 10)', 'R': '(177, 13)'},
    'self_portrait': {'G': '(78, 29)', 'R': '(176, 37)'},
    'siren': {'G': '(49, -6)', 'R': '(96, -24)'},
    'three_generations': {'G': '(54, 12)', 'R': '(111, 9)'},
    'tobolsk': {'G': '(3, 3)', 'R': '(6, 3)'},
    'monastery': {'G': '(-3, 2)', 'R': '(3, 2)'},
    'cathedral': {'G': '(5, 2)', 'R': '(12, 3)'}
}

def get_offset(filename):
    # Try different ways to match the filename to keys
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    
    # Handle _aligned suffix
    if name_no_ext.endswith('_aligned'):
        name_no_ext = name_no_ext.replace('_aligned', '')
        
    if name_no_ext in raw_offsets:
        return raw_offsets[name_no_ext]
    
    return {'G': 'N/A', 'R': 'N/A'}

# Image lists
jpegs = ['cathedral.jpg', 'monastery.jpg', 'tobolsk.jpg']

# Specific images requested by user (Updated with 3rd selection)
my_collection = [
    {
        'file': 'master-pnp-prok-00100-00172a_aligned.png',
        'title': 'Master PNP 00172a (Selection 1)'
    },
    {
        'file': 'master-pnp-prok-00100-00187a_aligned.png',
        'title': 'Master PNP 00187a (Selection 2)'
    },
    {
        'file': 'master-pnp-prok-00000-00082a_aligned.png', 
        'title': 'Master PNP 00082a (Selection 3)'
    }
]

# High res PNGs - filter out the 'my_collection' ones
all_pngs = [f for f in os.listdir(os.path.join(base_dir, out_dir_name)) if f.endswith('.png')]
blacklist = [m['file'] for m in my_collection]
valid_pngs = [f for f in all_pngs if f not in blacklist]
valid_pngs.sort()

# HTML Content
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prokudin-Gorskii Collection</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {{
            --primary-bg: #ffffff;
            --secondary-bg: #f3f6f9; /* Light gray-blue for alternate sections */
            --header-bg: #1a1a1a;
            --text-color: #2c3e50;
            --accent-blue: #3498db;
            --accent-red: #e74c3c;
            --accent-green: #2ecc71;
            --card-shadow: 0 10px 20px rgba(0,0,0,0.08);
            --nav-height: 60px;
        }}
        
        html {{
            scroll-behavior: smooth;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--primary-bg);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            line-height: 1.6;
            padding-top: var(--nav-height);
        }}

        /* Navigation */
        nav {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: var(--nav-height);
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid #eee;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .nav-links {{
            display: flex;
            gap: 30px;
        }}
        
        .nav-links a {{
            text-decoration: none;
            color: #555;
            font-weight: 500;
            font-size: 0.95rem;
            transition: color 0.3s;
            padding: 5px 10px;
            border-radius: 5px;
        }}
        
        .nav-links a:hover {{
            color: var(--accent-blue);
            background: #f0f7ff;
        }}

        /* Hero Header */
        header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            text-align: center;
            padding: 80px 20px;
            clip-path: polygon(0 0, 100% 0, 100% 85%, 0 100%);
            margin-bottom: 40px;
        }}
        
        header h1 {{
            font-size: 3rem;
            margin: 0;
            font-weight: 700;
            letter-spacing: -1px;
        }}
        
        .subtitle {{
            font-size: 1.1rem;
            font-weight: 500;
            opacity: 0.95;
            margin-top: 10px;
            color: #ecf0f1;
            background: rgba(255, 255, 255, 0.1);
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
        }}
        
        header p {{
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 20px auto 0;
        }}

        /* Sections */
        section {{
            padding: 60px 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .section-alt {{
            background-color: var(--secondary-bg);
            width: 100%;
            max-width: 100%; /* Full width background */
        }}
        .section-alt .content-wrapper {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}

        h2 {{
            font-size: 2rem;
            margin-bottom: 40px;
            color: #2c3e50;
            position: relative;
            display: inline-block;
        }}
        
        h2::after {{
            content: '';
            position: absolute;
            left: 0;
            bottom: -10px;
            width: 50px;
            height: 4px;
            background: var(--accent-blue);
            border-radius: 2px;
        }}

        /* Technical Implementation */
        .tech-grid {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 40px;
            align-items: start;
        }}
        
        .pipeline-step {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            border-left: 4px solid var(--accent-blue);
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: transform 0.2s;
        }}
        
        .pipeline-step:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        }}
        
        .step-num {{
            font-weight: 800;
            color: var(--accent-blue);
            margin-right: 10px;
        }}
        
        .params-card {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            position: sticky;
            top: 100px;
        }}
        
        .param-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding: 12px 0;
        }}
        
        .param-item:last-child {{ border-bottom: none; }}
        
        .param-key {{ font-family: monospace; font-weight: 600; color: #e67e22; }}
        .param-val {{ color: #7f8c8d; font-size: 0.9rem; }}

        /* Galleries */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 30px;
        }}
        
        .img-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            position: relative;
            top: 0;
        }}
        
        .img-card:hover {{
            top: -10px;
            box-shadow: 0 20px 30px rgba(0,0,0,0.12);
        }}
        
        .img-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .card-caption {{
            padding: 20px;
            border-top: 1px solid #f0f0f0;
        }}
        
        .card-title {{
            font-weight: 700;
            margin-bottom: 5px;
            color: #34495e;
        }}
        
        .offset-badge {{
            display: inline-block;
            background: #e8f4fc;
            color: var(--accent-blue);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-family: monospace;
            margin-top: 5px;
        }}
        
        .placeholder {{
            height: 250px;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #bdc3c7;
            font-style: italic;
        }}
        
        /* Badges for section headers */
        .section-badge {{
            background: #e74c3c;
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            vertical-align: middle;
            margin-left: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

    </style>
</head>
<body>

<nav>
    <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#implementation">Implementation</a>
        <a href="#results-jpeg">Single-Scale</a>
        <a href="#results-tiff">Multi-Scale</a>
        <a href="#my-collection">My Collection</a>
    </div>
</nav>

<header id="overview">
    <h1>Prokudin-Gorskii Collection</h1>
    <div class="subtitle">HW1 | COMS W 4732 CV2 | Prof. Aleksander Holynski</div>
    <p>Colorizing early 20th-century history using automated image alignment algorithms.</p>
</header>

<!-- Technical Section (Alt Background) -->
<div class="section-alt" id="implementation">
    <div class="content-wrapper">
        <section>
            <h2>Technical Implementation</h2>
            
            <div class="tech-grid">
                <!-- Pipeline List -->
                <div>
                    <h3 style="margin-bottom:20px;">Algorithm Pipeline</h3>
                    <div class="pipeline-step">
                        <span class="step-num">01</span>
                        <strong>Load & Preprocess</strong>
                        <div style="font-size:0.9rem; color:#666; margin-top:5px;">Check depth (16-bit vs 8-bit) and normalize to [0,1].</div>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-num">02</span>
                        <strong>Channel Splitting</strong>
                        <div style="font-size:0.9rem; color:#666; margin-top:5px;">Divide height by 3 to extract B, G, R channels.</div>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-num">03</span>
                        <strong>Feature Extraction</strong>
                        <div style="font-size:0.9rem; color:#666; margin-top:5px;">Compute Sobel edges to ignore intensity variations.</div>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-num">04</span>
                        <strong>Pyramid Alignment</strong>
                        <div style="font-size:0.9rem; color:#666; margin-top:5px;">Recursive alignment: Coarse (low-res) &#8594; Fine (high-res).</div>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-num">05</span>
                        <strong>Reconstruction</strong>
                        <div style="font-size:0.9rem; color:#666; margin-top:5px;">Shift original high-quality channels using calculated offsets.</div>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-num">06</span>
                        <strong>Post-Processing</strong>
                        <div style="font-size:0.9rem; color:#666; margin-top:5px;">Auto-crop borders (10%) and contrast stretching.</div>
                    </div>
                </div>
                
                <!-- Parameters Card -->
                <div class="params-card">
                    <h3 style="margin-top:0;">Configuration</h3>
                    <div class="param-item">
                        <span class="param-key">search_range</span>
                        <span class="param-val">10 px</span>
                    </div>
                    <div class="param-item">
                        <span class="param-key">pyramid_depth</span>
                        <span class="param-val">5 levels</span>
                    </div>
                    <div class="param-item">
                        <span class="param-key">crop_ratio</span>
                        <span class="param-val">0.5 (50%?)</span>
                    </div>
                    <div class="param-item">
                        <span class="param-key">metric</span>
                        <span class="param-val">SSD</span>
                    </div>
                </div>
            </div>
        </section>
    </div>
</div>

<!-- Single Scale Results -->
<section id="results-jpeg">
    <h2>Single-Scale Results <span class="section-badge">JPEG</span></h2>
    <div class="gallery-grid">
"""

for jpg in jpegs:
    img_path = urllib.parse.quote(f"{out_dir_name}/{jpg}")
    offs = get_offset(jpg)
    html += f"""
        <div class="img-card">
            <img src="{img_path}" alt="{jpg}">
            <div class="card-caption">
                <div class="card-title">{jpg}</div>
                <div class="offset-badge">G: {offs['G']}</div>
                <div class="offset-badge">R: {offs['R']}</div>
            </div>
        </div>
    """

html += """
    </div>
</section>

<!-- Multi Scale Results (Alt Background) -->
<div class="section-alt" id="results-tiff">
    <div class="content-wrapper">
        <section>
            <h2>Multi-Scale Results <span class="section-badge" style="background:#3498db;">TIFF</span></h2>
            <div class="gallery-grid">
"""

for png in valid_pngs:
    img_path = urllib.parse.quote(f"{out_dir_name}/{png}")
    name = png.replace('_aligned.png', '').title()
    offs = get_offset(png)
    html += f"""
                <div class="img-card">
                    <img src="{img_path}" alt="{name}">
                    <div class="card-caption">
                        <div class="card-title">{name}</div>
                        <div class="offset-badge">G: {offs['G']}</div>
                        <div class="offset-badge">R: {offs['R']}</div>
                    </div>
                </div>
    """

html += """
            </div>
        </section>
    </div>
</div>

<!-- My Collection -->
<section id="my-collection">
    <h2>My Collection</h2>
    <div class="gallery-grid">
"""

for item in my_collection:
    img_path = f"{out_dir_name}/{item['file']}"
    title = item['title']
    full_path = os.path.join(base_dir, out_dir_name, item['file'])
    offs = get_offset(item['file'])
    
    if os.path.exists(full_path):
        html += f"""
            <div class="img-card">
                <img src="{urllib.parse.quote(img_path)}" alt="{title}">
                <div class="card-caption">
                    <div class="card-title">{title}</div>
                    <div class="offset-badge">G: {offs['G']}</div>
                    <div class="offset-badge">R: {offs['R']}</div>
                </div>
            </div>
        """
    else:
        html += f"""
            <div class="img-card">
                <div class="placeholder">Image Pending</div>
                <div class="card-caption">
                    <div class="card-title">{title}</div>
                    <div class="offset-badge">G: {offs['G']}</div>
                    <div class="offset-badge">R: {offs['R']}</div>
                </div>
            </div>
        """

html += """
    </div>
</section>

<footer style="text-align:center; padding:40px; color:#95a5a6; font-size:0.9rem;">
    &copy; {year} Computer Vision HW1
</footer>

</body>
</html>
""".format(year=datetime.now().year)

with open(os.path.join(base_dir, 'report.html'), 'w') as f:
    f.write(html)

print("Report generated successfully: report.html")
