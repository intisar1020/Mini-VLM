from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


# class PaliGemmaForConditionalGeneration(nn.Module):
#     def __init__(self, config: PaliGemmaConfig):
#         super().__init__()
        


config = SiglipVisionConfig()
model = SiglipVisionModel(config=config)
print (model)