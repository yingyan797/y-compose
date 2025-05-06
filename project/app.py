from flask import Flask, render_template, request, jsonify, send_file
import json, os, torch, io, base64
import numpy as np
from PIL import Image
from datetime import datetime
from ltl_util import formula_to_dfa, parse_dfa

app = Flask(__name__)

# Directory to store saved grids
GRIDS_DIR = 'project/static/saved_grids'
MONA_DIR = 'project/static/mona_files'
DFA_DIR = 'project/static/dfa_files'

# Create directory if it doesn't exist
if not os.path.exists(GRIDS_DIR):
    os.makedirs(GRIDS_DIR)

@app.route('/')
def index():
    """Render the main page with empty grid"""
    return render_template('index.html')

@app.route('/get_images/<fname>')
def get_images(fname:str):
    project = torch.load(f"project/static/saved_cont/{fname}")
    images = []
    for layer in project["terrain"]:  # Creating 3 sample images
        # Create a sample numpy array (100x100 with random values)
        mask: torch.IntTensor = layer["mask"]
        hasimage = torch.any(mask).item() and not torch.all(mask).item()
        img_str = ""
        if hasimage:
            color = layer["color"][1:]
            rgba = torch.IntTensor([int(c,16) for c in [color[:2], color[2:4], color[4:]]]+[255])
            # mask = torch.mul(mask, torch.IntTensor(rgb))
            black = torch.zeros(mask.shape, dtype=torch.int).unsqueeze(2).repeat(1,1,4)
            hascolor = torch.nonzero(mask)
            rows, cols = hascolor[:,0], hascolor[:,1]
            # Convert numpy array to image
            black[rows, cols] = rgba
            img = Image.fromarray(black.numpy().astype(np.uint8))
            # Convert to base64 string
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        images.append({"name":layer["name"], "hasimage": hasimage, "image":img_str})
    
    return jsonify({'images': images})

@app.route('/save_images/<fname>', methods=['POST'])
def save_images(fname):
    # Get the image data from the request
    layers = request.json.get('terrain')
    terrain = []
    # Convert base64 back to image
    for layer in layers:
        img_data = base64.b64decode(layer["mask"])
        img = torch.IntTensor(np.array(Image.open(io.BytesIO(img_data))))
        img = img[:,:,3].to(torch.bool)
        terrain.append({"name": layer["name"], "color": layer["color"], "mask": img})
    project = {"dim": request.json.get('dim'), "terrain": terrain}
    torch.save(project, f"project/static/saved_cont/{fname}")
    
    return jsonify({
        'success': True,
        'message': f'Terrain saved successfully as {fname}',
    })
        


@app.route('/ltl')
def ltl_formulation():
    files = [fname[:-5] for fname in os.listdir(MONA_DIR)]
    return render_template("ltl.html", files=files)

@app.route('/create_dfa', methods=["POST"])
def create_dfa():
    """Load grid data from a saved file"""
    try:
        ifml = request.form.get("formula")
        fname = request.form.get("fname")
        out = formula_to_dfa(ifml, fname)
        if isinstance(out, dict):
            res = {'success': True}
            res.update(out)
            return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_dfa/<fname>')
def load_dfa(fname):
    try:
        with open(MONA_DIR+f"/{fname}.mona", "r") as f:
            formula = f.readline().split(";")[0][1:]
        with open(DFA_DIR+f"/{fname}.dfa", "r") as f:
            mona_out = f.read()
    except IOError as e:
        return jsonify({'success': False, 'error': str(e)})
    res = {"success": True}
    res.update(parse_dfa(formula, mona_out))
    return jsonify(res)

@app.route('/save_grid', methods=['POST'])
def save_grid():
    """Save grid data to a JSON file"""
    
    return jsonify({'success': False, 'error': str(e)})

@app.route('/load_grid/<grid_id>')
def load_grid(grid_id):
    """Load grid data from a saved file"""

    return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_grid/<grid_id>', methods=['POST'])
def delete_grid(grid_id):
    """Delete a saved grid"""
    try:
        filename = os.path.join(GRIDS_DIR, f"{grid_id}.json")
        
        if os.path.exists(filename):
            os.remove(filename)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Grid not found'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run("0.0.0.0", 5002, debug=True)