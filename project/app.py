from flask import Flask, render_template, request, jsonify, send_file
import json, os, torch, io, base64
import numpy as np
from PIL import Image
from datetime import datetime
from ltl_util import formula_to_dfa, parse_dfa, output2dot

app = Flask(__name__)

# Directory to store saved grids
GRIDS_DIR = 'project/static/saved_disc'
IMAGES_DIR = 'project/static/saved_cont'
MONA_DIR = 'project/static/mona_files'
DFA_DIR = 'project/static/dfa_files'
ATASK_DIR = 'project/static/atomic_tasks'

def color_rgb(ccode):
    color = ccode[1:]
    return [int(c,16) for c in [color[:2], color[2:4], color[4:]]]

@app.route('/')
def index():
    """Render the main page with empty grid"""
    return render_template('index.html')

@app.route('/get_images', methods=["POST"])
def get_images():
    try:
        project = torch.load(f"{IMAGES_DIR}/{request.json.get('fname')}")
    except FileNotFoundError:
        return jsonify({"success": False})
    images = []
    for layer in project["terrain"]:  # Creating 3 sample images
        # Create a sample numpy array (100x100 with random values)
        mask: torch.IntTensor = layer["mask"]
        hasimage = torch.any(mask).item()
        img_str = ""
        if hasimage:
            # mask = torch.mul(mask, torch.IntTensor(rgb))
            black = torch.zeros(mask.shape, dtype=torch.int).unsqueeze(2).repeat(1,1,4)
            hascolor = torch.nonzero(mask)
            rows, cols = hascolor[:,0], hascolor[:,1]
            # Convert numpy array to image
            black[rows, cols] = torch.IntTensor(layer["color"]+[255])
            img = Image.fromarray(black.numpy().astype(np.uint8))
            # Convert to base64 string
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        images.append({"name":layer["name"], "always": layer["always"], "hasimage": hasimage, "image":img_str})
    
    return jsonify({'success':True, 'images': images})

@app.route('/save_images', methods=['POST'])
def save_images():
    # Get the image data from the request
    layers = request.json.get('terrain')
    terrain = []
    # Convert base64 back to image
    for layer in layers:
        img_data = base64.b64decode(layer["mask"])
        img = torch.IntTensor(np.array(Image.open(io.BytesIO(img_data))))
        img = img[:,:,3].to(torch.bool)
        terrain.append({"name": layer["name"], "color": color_rgb(layer["color"]), "always": layer["always"], "mask": img})
    project = {"dim": request.json.get('dim'), "terrain": terrain}
    fname = request.json.get("fname")
    torch.save(project, f"{IMAGES_DIR}/{fname}")
    
    return jsonify({
        'success': True,
        'message': f'Terrain saved successfully as {fname}',
    })

@app.route('/save_grid', methods=['POST'])
def save_grid():
    """Save grid data to a pt file"""
    dim = request.json.get("dim")
    project = {"dim": dim}
    terrain = []
    for layer in request.json.get("terrain"):
        mask = torch.zeros(dim[0], dim[1], dtype=torch.bool)
        locs = torch.IntTensor([[int(i) for i in k.split(",")] for k,v in layer["mask"].items() if v])
        if locs.shape[0] > 0:
            mask[locs[:,0], locs[:,1]] = 1
        terrain.append({"name": layer["name"], "color": color_rgb(layer["color"]), "always": layer["always"], "mask": mask})
    project = {"dim": request.json.get('dim'), "terrain": terrain}
    fname = request.json.get("fname")
    torch.save(project, f"{GRIDS_DIR}/{fname}")
    
    return jsonify({
        'success': True,
        'message': f'Terrain saved successfully as {fname}',
    })
    

@app.route('/load_grid', methods=['POST'])
def load_grid():
    """Load grid data from a saved file"""
    filename = f"{GRIDS_DIR}/{request.json.get('fname')}"
    try:
        project = torch.load(filename)
    except FileNotFoundError:
        return jsonify({"success": False})
    grid = []
    for layer in project["terrain"]:  # Creating 3 sample images
        # Create a sample numpy array (100x100 with random values)
        mask: torch.BoolTensor = layer["mask"]
        hasimage = torch.any(mask).item()
        data = {}
        if hasimage:
            for loc in torch.nonzero(mask).numpy().tolist():
                data[f"{loc[0]},{loc[1]}"] = True
        grid.append({"name":layer["name"], "always": layer["always"], "hasimage": hasimage, "data":data})
    return jsonify({"success":True, "rows": project["dim"][0], "cols":project["dim"][1], "grid": grid})

@app.route('/deltrn/<mode>', methods=['PUT'])
def deltrn(mode):
    """Delete a saved terrain"""
    filename = f"project/static/{mode}/{request.json.get('fname')}"
    if os.path.exists(filename):
        os.remove(filename)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'File not found'})

@app.route('/ltl')
def ltl_formulation():
    return render_template("ltl.html")

@app.route('/create_dfa', methods=["POST"])
def create_dfa():
    """Load grid data from a saved file"""
    ifml = request.json.get("formula")
    fname = request.json.get("fname")
    out = formula_to_dfa(ifml, fname)
    if isinstance(out, tuple):
        res = {'success': True, "mona": out[1]}
        res.update(out[0])
        if (request.json.get("visual")):
            res["diagram_code"] = output2dot(out[2])
        return jsonify(res)

@app.route('/load_dfa', methods=["POST"])
def load_dfa():
    fname = request.json.get("fname")
    try:
        with open(MONA_DIR+f"/{fname}.mona", "r") as f:
            lines = f.readlines()
            formula = lines[0][1:-2]
            mona_in = lines[1:]
        with open(DFA_DIR+f"/{fname}.dfa", "r") as f:
            mona_out = f.read()
    except IOError as e:
        return jsonify({'success': False, 'error': str(e)})
    res = {"success": True, "mona": mona_in}
    res.update(parse_dfa(formula, mona_out))
    if (request.json.get("visual")):
        res["diagram_code"] = output2dot(mona_out)
    return jsonify(res)

@app.route('/del_dfa', methods=["POST"])
def del_dfa():
    fname = request.json.get("fname")
    mona_path = MONA_DIR+f"/{fname}.mona"
    dfa_path = DFA_DIR+f"/{fname}.dfa"
    if os.path.isfile(mona_path):
        os.remove(mona_path)
    if os.path.isfile(dfa_path):
        os.remove(dfa_path)
    
    return jsonify({'success': True})

@app.route('/create_atask', methods=['POST'])
def create_atask():
    fname = request.json.get("fname")
    with open(ATASK_DIR+f"/{fname}.t", "w") as f:
        for i, atask in enumerate(request.json.get("tasks")[3:], start=1):
            f.write(f"$task_{i} = '{atask}'\n")
    
    return jsonify({'success': True})

@app.route('/load_atask', methods=['POST'])
def load_atask():
    fname = request.json.get("fname")
    with open(ATASK_DIR+f"/{fname}", "r") as f:
        tasks = []
        while True:
            line = f.readline()
            if not line:
                break
            for i in range(1, len(line)):
                if not line[i].isalnum() and line[i] not in ["_"]:
                    tname = line[1:i]
                    break
            else:
                tname = line.strip()
            tasks.append(tname)
    return jsonify({'success': True, "tasks": tasks})

if __name__ == '__main__':
    app.run("127.0.0.1", 5002, debug=True)