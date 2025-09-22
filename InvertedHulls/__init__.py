import os
from Malt.PipelinePlugin import PipelinePlugin, isinstance_str

class InvertedHullsPlugin(PipelinePlugin):

    @classmethod
    def poll_pipeline(self, pipeline):
        return isinstance_str(pipeline, 'NPR_Pipeline')
    
    @classmethod
    def register_graph_libraries(self, graphs):
        root = os.path.dirname(__file__)
        graphs['Render'].add_library(os.path.join(root, 'Render'))
        graphs['Mesh'].add_library(os.path.join(root, 'Shaders'))

PLUGIN = InvertedHullsPlugin
