from os.path import expanduse
import 
from data_utils.pathways.gmt_pathway import get_KEGG_map

input_genes = ['AR', 'AKT', 'EGFR']
filename = expanduser('../../_database/pathways/MsigDB/c2.cp.kegg.v6.1.symbols.gmt')
mapp, genes, pathways = get_KEGG_map(input_genes, filename)
print('genes', genes)
print('pathways', pathways)
print('mapp', mapp)
