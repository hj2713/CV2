
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

# Settings
PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754 # A4 at ~150 DPI
MARGIN = 100
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
OUTPUT_FILENAME = "HW1_Report.pdf"

# Colors
COLOR_TEXT = "#2c3e50"
COLOR_ACCENT = "#3498db"
COLOR_BG = "#ffffff"

# Offsets (Copied from generate_report.py)
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
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    if name_no_ext.endswith('_aligned'):
        name_no_ext = name_no_ext.replace('_aligned', '')
    return raw_offsets.get(name_no_ext, {'G': 'N/A', 'R': 'N/A'})

def create_page():
    return Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), COLOR_BG)

def load_fonts():
    try:
        title_font = ImageFont.truetype(FONT_PATH, 60)
        heading_font = ImageFont.truetype(FONT_PATH, 40)
        body_font = ImageFont.truetype(FONT_PATH, 24)
        caption_font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        print("Warning: Could not load Arial Unicode. Using default font.")
        title_font = ImageFont.load_default()
        heading_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()
    return title_font, heading_font, body_font, caption_font

def draw_wrapped_text(draw, text, font, color, x, y, max_width):
    lines = textwrap.wrap(text, width=int(max_width / (font.getbbox("A")[2] if hasattr(font, "getbbox") else 10))) # Rough calc
    # Better wrap using pixel width
    lines = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Check width
            bbox = font.getbbox(test_line) if hasattr(font, "getbbox") else (0,0, len(test_line)*10, 20)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
    
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=color)
        bbox = font.getbbox("Tg") if hasattr(font, "getbbox") else (0,0,0,25)
        h = bbox[3] - bbox[1]
        current_y += h * 1.5
    return current_y

def main():
    print("Generating PDF Report...")
    title_font, heading_font, body_font, caption_font = load_fonts()
    pages = []
    
    # --- Page 1: Cover ---
    page1 = create_page()
    draw = ImageDraw.Draw(page1)
    
    # Title
    draw.text((PAGE_WIDTH//2, 400), "Prokudin-Gorskii Collection", font=title_font, fill=COLOR_TEXT, anchor="ms")
    draw.text((PAGE_WIDTH//2, 500), "HW1 | Computer Vision", font=heading_font, fill=COLOR_TEXT, anchor="ms")
    
    # Submitted By
    draw.text((PAGE_WIDTH//2, 600), "Submitted By:", font=body_font, fill="#7f8c8d", anchor="ms")
    draw.text((PAGE_WIDTH//2, 650), "Himanshu Jhawar (hj2713)", font=heading_font, fill=COLOR_TEXT, anchor="ms")
    
    # GitHub
    draw.text((PAGE_WIDTH//2, 750), "GitHub Repository:", font=body_font, fill="#7f8c8d", anchor="ms")
    draw.text((PAGE_WIDTH//2, 800), "https://github.com/hj2713/CV2/tree/main/HW1", font=body_font, fill=COLOR_ACCENT, anchor="ms")

    # Add a hero image if available
    hero_path = "output_images/icon_aligned.png" 
    if os.path.exists(hero_path):
        try:
            hero = Image.open(hero_path)
            # Resize to fit width
            target_w = PAGE_WIDTH - 2*MARGIN
            ratio = target_w / hero.width
            target_h = int(hero.height * ratio)
            
            # Crop height if too tall
            if target_h > 600:
                target_h = 600
                hero = hero.resize((target_w, int(hero.height * ratio)), Image.Resampling.LANCZOS)
                hero = hero.crop((0, 0, target_w, target_h))
            else:
                 hero = hero.resize((target_w, target_h), Image.Resampling.LANCZOS)

            page1.paste(hero, (MARGIN, 900))
        except Exception as e:
            print(f"Failed to load hero image: {e}")
            
    pages.append(page1)
    
    # --- Page 2: Implementation ---
    page2 = create_page()
    draw = ImageDraw.Draw(page2)
    y = MARGIN
    
    draw.text((MARGIN, y), "Technical Implementation", font=heading_font, fill=COLOR_TEXT)
    y += 80
    
    impl_text = """
    Algorithm Pipeline:
    
    1. Load & Preprocess: Images are loaded (checking for 16-bit) and normalized.
    
    2. Channel Splitting: The vertical glass-plate image is split into equal thirds (B, G, R).
    
    3. Feature Extraction: Sobel edges are computed for each channel. This makes the alignment robust to brightness differences between channels.
    
    4. Pyramid Alignment: A 5-level image pyramid is used. Alignment starts at the coarsest level (smallest image) to find approximate offsets, then refines them at each higher resolution.
    
    5. Reconstruction: The Red and Green channels are shifted using the calculated (x, y) offsets to align with the Blue channel.
    
    6. Post-Processing: Borders are auto-cropped (10%) to remove artifacts, and contrast is stretched for better visibility.
    """
    
    y = draw_wrapped_text(draw, impl_text, body_font, COLOR_TEXT, MARGIN, y, PAGE_WIDTH - 2*MARGIN)
    
    pages.append(page2)
    
    # --- Page 3+: Gallery ---
    # Collect all images
    image_files = []
    # JPEGs
    for jpg in ['cathedral.jpg', 'monastery.jpg', 'tobolsk.jpg']:
        image_files.append(('output_images/' + jpg, jpg))
    # PNGs
    for f in sorted(os.listdir('output_images')):
        if f.endswith('.png'):
             # Determine if it's 'my selection'
             is_selection = False
             # Simple filter: include all aligned pngs
             if '_aligned' in f:
                 image_files.append(('output_images/' + f, f.replace('_aligned.png', '').title()))

    # Grid Layout Settings
    cols = 2
    rows = 2
    items_per_page = cols * rows
    cell_width = (PAGE_WIDTH - 2*MARGIN - 50) // cols
    cell_height = 600 # Fixed height allocation
    
    current_page = None
    draw = None
    
    for i, (path, title) in enumerate(image_files):
        if i % items_per_page == 0:
            current_page = create_page()
            draw = ImageDraw.Draw(current_page)
            draw.text((MARGIN, MARGIN/2), f"Results Gallery ({i//items_per_page + 1})", font=heading_font, fill=COLOR_TEXT)
            pages.append(current_page)
            
        # Determine Grid Position
        pos_index = i % items_per_page
        row = pos_index // cols
        col = pos_index % cols
        
        x = MARGIN + col * (cell_width + 50)
        y = MARGIN + 100 + row * (cell_height + 50)
        
        # Load and resize image
        if os.path.exists(path):
            try:
                img = Image.open(path)
                # Keep aspect ratio
                ratio = min(cell_width / img.width, (cell_height - 100) / img.height)
                new_w = int(img.width * ratio)
                new_h = int(img.height * ratio)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Center in cell
                img_x = x + (cell_width - new_w) // 2
                current_page.paste(img, (img_x, y))
                
                # Draw Caption
                offs = get_offset(path)
                caption = f"{title}\nG: {offs['G']}, R: {offs['R']}"
                draw.text((x, y + new_h + 10), caption, font=caption_font, fill=COLOR_TEXT)
                
            except Exception as e:
                print(f"Error drawing {path}: {e}")
    
    # Save PDF
    if pages:
        pages[0].save(OUTPUT_FILENAME, save_all=True, append_images=pages[1:], resolution=150)
        print(f"Successfully generated {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()
