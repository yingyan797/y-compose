from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from datetime import datetime

app = Flask(__name__)

# Directory to store saved grids
GRIDS_DIR = 'project/static/saved_grids'

# Create directory if it doesn't exist
if not os.path.exists(GRIDS_DIR):
    os.makedirs(GRIDS_DIR)

@app.route('/')
def index():
    """Render the main page with empty grid"""
    # Get list of saved grids
    saved_grids = get_saved_grids()
    return render_template('index.html', saved_grids=saved_grids)

@app.route('/save_grid', methods=['POST'])
def save_grid():
    """Save grid data to a JSON file"""
    try:
        # Get data from form
        grid_rows = int(request.form.get('grid_rows', 8))
        grid_cols = int(request.form.get('grid_cols', 8))
        grid_name = request.form.get('grid_name', 'Untitled Grid')
        selected_cells = json.loads(request.form.get('selected_cells', '[]'))
        cell_types = json.loads(request.form.get('cell_types', '{}'))
        
        # Create grid data object
        grid_data = {
            'grid_name': grid_name,
            'grid_rows': grid_rows,
            'grid_cols': grid_cols,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'selected_cells': selected_cells,
            'cell_types': cell_types
        }
        
        # Determine if this is an update or new grid based on hidden input
        grid_id = request.form.get('grid_id')
        
        if not grid_id or request.form.get("copy") == "true":
            # Generate new ID for the grid
            grid_id = datetime.now().strftime('%Y%m%d%H%M%S')
            grid_data['id'] = grid_id
        else:
            # Keep existing ID for updates
            grid_data['id'] = grid_id
            # Update the timestamp
            grid_data['updated_at'] = datetime.now().isoformat()
        
        # Save to file
        filename = os.path.join(GRIDS_DIR, f"{grid_id}.json")
        with open(filename, 'w') as f:
            json.dump(grid_data, f, indent=2)
        
        return jsonify({
            'success': True, 
            'grid_id': grid_id,
            'grid_name': grid_name,
            'grid_rows': grid_rows,
            'grid_cols': grid_cols,
            'message': 'Grid saved successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_grid/<grid_id>')
def load_grid(grid_id):
    """Load grid data from a saved file"""
    try:
        filename = os.path.join(GRIDS_DIR, f"{grid_id}.json")
        
        if not os.path.exists(filename):
            return jsonify({'success': False, 'error': 'Grid not found'})
        
        with open(filename, 'r') as f:
            grid_data = json.load(f)
        
        return jsonify({
            'success': True, 
            'grid_rows': grid_data.get('grid_rows', 8),
            'grid_cols': grid_data.get('grid_cols', 8),
            'grid_name': grid_data.get('grid_name', 'Untitled Grid'),
            'selected_cells': grid_data.get('selected_cells', []),
            'cell_types': grid_data.get('cell_types', {}),
            'grid_id': grid_id
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
'''
@app.route('/grid/<grid_id>')
def view_grid(grid_id):
    """Render page with existing grid loaded"""
    try:
        filename = os.path.join(GRIDS_DIR, f"{grid_id}.json")
        
        if not os.path.exists(filename):
            return redirect(url_for('index'))
        
        with open(filename, 'r') as f:
            grid_data = json.load(f)
        
        # Get list of saved grids for the dropdown
        saved_grids = get_saved_grids()
        
        # Extract all custom types (those not in the default set)
        default_types = {'obstacle', 'goal-1', 'goal-2', 'goal-3', 'goal-4', 'goal-5'}
        custom_types = {key: value for key, value in grid_data.get('cell_types', {}).items() 
                       if key not in default_types}
        
        return render_template('index.html',
                              grid_rows=grid_data.get('grid_rows', 8),
                              grid_cols=grid_data.get('grid_cols', 8),
                              grid_name=grid_data.get('grid_name', ''),
                              grid_id=grid_id,
                              selected_cells=json.dumps(grid_data.get('selected_cells', [])),
                              custom_types=custom_types,
                              saved_grids=saved_grids)
    
    except Exception as e:
        print(f"Error loading grid: {e}")
        return redirect(url_for('index'))
'''

def get_saved_grids():
    """Get list of all saved grids for dropdown"""
    grids = []
    
    if os.path.exists(GRIDS_DIR):
        for filename in os.listdir(GRIDS_DIR):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(GRIDS_DIR, filename)
                    with open(filepath, 'r') as f:
                        grid_data = json.load(f)
                    
                    grids.append({
                        'id': grid_data.get('id', filename.replace('.json', '')),
                        'name': grid_data.get('grid_name', 'Untitled Grid'),
                        'rows': grid_data.get('grid_rows', 8),
                        'cols': grid_data.get('grid_cols', 8),
                        'updated_at': grid_data.get('updated_at', '')
                    })
                except Exception as e:
                    print(f"Error loading grid data from {filename}: {e}")
    
    # Sort by update time (newest first)
    grids.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return grids

    
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