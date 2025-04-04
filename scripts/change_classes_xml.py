import os
from xml.etree import ElementTree as ET

"""
    批量修改xml文件中的类别
    将'lymph', 'neutro', 'histo', 'meso'替换为'normal'
"""

def modify_xml_categories(source_directory, target_directory):
    # Categories to be replaced with 'normal'
    categories_to_replace = {'lymph', 'neutro', 'histo', 'meso'}
    
    # Ensure target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Iterate over all XML files in the provided source directory
    for filename in os.listdir(source_directory):
        if not filename.endswith('.xml'):
            continue  # Skip non-XML files

        source_path = os.path.join(source_directory, filename)
        target_path = os.path.join(target_directory, filename)
        
        # Load and parse the XML file
        tree = ET.parse(source_path)
        root = tree.getroot()

        # Iterate over each 'object' in the XML and update 'name' if it matches one of the specified categories
        for obj in root.findall('object'):
            name = obj.find('name')
            if name.text in categories_to_replace:
                name.text = 'normal'
        
        # Save the modified XML to the target directory
        tree.write(target_path)
        print(f'Modified and saved to new location: {filename}')

# Example usage
source_directory = 'change_classes/xml_train'
target_directory = 'change_classes/xml_new_train'
modify_xml_categories(source_directory, target_directory)
