from csp import CSP, BinaryConstraint
from typing import List, Tuple, Dict, Optional

class MapColoringProblem:
    """Map Coloring CSP Problem that wraps integer-based CSP with human-readable interface"""
    
    def __init__(self, regions: List[str], neighbors: List[Tuple[str, str]], colors: List[str]):
        self.regions = regions
        self.neighbors = neighbors
        self.colors = colors
        
        # Create mappings between human-readable and integer representations
        self.region_to_index = {region: i for i, region in enumerate(regions)}
        self.index_to_region = {i: region for i, region in enumerate(regions)}
        self.color_to_index = {color: i for i, color in enumerate(colors)}
        self.index_to_color = {i: color for i, color in enumerate(colors)}
    
    def create_csp(self) -> CSP:
        num_variables = len(self.regions)
        domains = [list(range(len(self.colors))) for _ in range(num_variables)]
        
        csp = CSP(num_variables, domains)
        
        # Add constraints for neighboring regions
        for region1, region2 in self.neighbors:
            var1 = self.region_to_index[region1]
            var2 = self.region_to_index[region2]
            
            def not_equal_constraint(color_idx1, color_idx2):
                return color_idx1 != color_idx2
            
            csp.add_constraint(BinaryConstraint(var1, var2, not_equal_constraint))
        
        return csp
    
    def translate_solution(self, int_solution: Optional[List[int]]) -> Optional[Dict[str, str]]:
        """Translate integer-based solution back to human-readable format"""
        if int_solution is None:
            return None
        
        solution = {}
        for var_idx, color_idx in enumerate(int_solution):
            if color_idx is not None:
                region = self.index_to_region[var_idx]
                color = self.index_to_color[color_idx]
                solution[region] = color
        
        return solution

# Australia map coloring problem
def australia_map_coloring():
    regions = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    neighbors = [
        ('WA', 'NT'), ('WA', 'SA'),
        ('NT', 'SA'), ('NT', 'Q'),
        ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
        ('Q', 'NSW'),
        ('NSW', 'V')
    ]
    colors = ['red', 'green', 'blue']
    
    return MapColoringProblem(regions, neighbors, colors)

# United States map coloring (simplified)
def usa_map_coloring():
    regions = ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM']
    neighbors = [
        ('WA', 'OR'), ('WA', 'ID'),
        ('OR', 'CA'), ('OR', 'ID'), ('OR', 'NV'),
        ('CA', 'NV'), ('CA', 'AZ'),
        ('NV', 'ID'), ('NV', 'UT'), ('NV', 'AZ'),
        ('ID', 'MT'), ('ID', 'WY'), ('ID', 'UT'),
        ('MT', 'WY'),
        ('WY', 'UT'), ('WY', 'CO'),
        ('UT', 'CO'), ('UT', 'AZ'), ('UT', 'NM'),
        ('CO', 'NM'),
        ('AZ', 'NM')
    ]
    colors = ['red', 'green', 'blue', 'yellow']
    
    return MapColoringProblem(regions, neighbors, colors)