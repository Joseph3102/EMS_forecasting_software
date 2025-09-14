import sys
import os
from py2puml.py2puml import py2puml

project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src', 'main')

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Generate UML for everything under src/main
puml_content = ''.join(py2puml(src_path, 'src.main'))

os.makedirs('docs', exist_ok=True)
with open('docs/diagram.puml', 'w') as f:
    f.write(puml_content)

print("UML diagram generated successfully in docs/diagram.puml")
