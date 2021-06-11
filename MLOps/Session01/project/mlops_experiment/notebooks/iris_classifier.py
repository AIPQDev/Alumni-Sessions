from bentoml import env, artifacts, api, BentoService, web_static_content
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact

@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
@web_static_content('./static')
class IrisClassifier(BentoService):

    @api(input=DataframeInput(), batch=True)
    def test(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
