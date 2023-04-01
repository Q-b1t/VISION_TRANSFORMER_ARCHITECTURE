
import torch
import torch.nn as nn

class PatcherModule(nn.Module):
  def __init__(self,img_shape,patch_size = 16,embedding_dim=768):
    super().__init__()
    # patcher hyperparameters
    self.batch_size,self.n_channels,self.height,self.width = img_shape
    self.patch_size = patch_size
    self.embedding_dim = embedding_dim
    self.patch_num = int((self.height * self.width)/self.patch_size ** 2)

    # gets the feature maps out of a image
    self.feature_mapper = nn.Conv2d(
        in_channels = self.n_channels,
        out_channels = self.embedding_dim,
        kernel_size = self.patch_size,
        padding = 0,
        stride = self.patch_size
    )

    # create a layer to flatten the feature map into a sequence of vectors
    self.flattener = nn.Flatten(
        start_dim = 2,
        end_dim = 3
    )

    self.class_token = nn.Parameter(
        data = torch.rand(
            self.batch_size,1,self.embedding_dim
        ),
        requires_grad = True
    )

    self.positional_encoding = nn.Parameter(
        data = torch.rand(
            1,self.patch_num + 1,self.embedding_dim
        ),
        requires_grad = True
    )


  def forward(self,x):
    # forward pass
    feature_maps = self.feature_mapper(x)

    flattened_sequence = self.flattener(feature_maps).permute(0, 2, 1) 

    patch_embedding_class_token = torch.cat([self.class_token,flattened_sequence],dim = 1)

    patch_and_position_embedding = patch_embedding_class_token + self.positional_encoding
    return patch_and_position_embedding



class MultiheadSelfAttentionModule(nn.Module):
  def __init__(self,embedding_dim = 768,num_heads = 12,attn_dropout = 0):
    super().__init__()
    # module hyperparameter
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.attn_dropout = attn_dropout

    # instance layers
    self.layer_norm = nn.LayerNorm(
        normalized_shape=self.embedding_dim
    )

    # instance miltihead attention
    self.multihead_attn = nn.MultiheadAttention(
        embed_dim = self.embedding_dim,
        num_heads = self.num_heads,
        dropout = self.attn_dropout,
        batch_first=True
    )

  def forward(self,x):
    x = self.layer_norm(x)
    attn_output,_ = self.multihead_attn(
        query = x,
        key = x,
        value = x,
        need_weights = False
    )
    return attn_output


class MultilayerPerceptronModule(nn.Module):
  def __init__(self,embedding_dim = 768,mlp_size = 3072,dropout = 0.1):
    super().__init__()
    
    # instance parameters
    self.embedding_dim = embedding_dim
    self.mlp_size = mlp_size
    self.dropout_rate = dropout

    # create the norm layer (LN)
    self.layer_norm = nn.LayerNorm(
        normalized_shape = self.embedding_dim
    )

    # create the multilayer perceptron
    self.mpl = nn.Sequential(
        nn.Linear(
            in_features=self.embedding_dim,
            out_features = self.mlp_size
        ),
        nn.GELU(),
        nn.Dropout(
            p = self.dropout_rate
        ),
        nn.Linear(
            in_features = self.mlp_size,
            out_features = self.embedding_dim
        ),
        nn.Dropout(
            p = self.dropout_rate
        )
    )

  def forward(self,x):
    return self.mpl(self.layer_norm(x)) # operator fussion



class TransfomerEncoderModule(nn.Module):
  def __init__(self,embedding_dim = 768,num_heads = 12, mlp_size = 3072,mlp_dropout = 0.1,attn_dropout = 0):
    super().__init__()
    # model hyperparemeter
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.mlp_size = mlp_size
    self.mlp_dropout = mlp_dropout
    self.attn_dropout = attn_dropout

    # instance model
    self.msa_block = MultiheadSelfAttentionModule(
        embedding_dim = self.embedding_dim,
        num_heads = self.num_heads,
        attn_dropout= self.attn_dropout
    )

    # create mpl block
    self.mpl_block = MultilayerPerceptronModule(
        embedding_dim= self.embedding_dim,
        mlp_size=self.mlp_size,
        dropout=self.mlp_dropout
    )

  def forward(self,x):
    return self.mpl_block(self.msa_block(x)+x)+x




class Vit(nn.Module):
  def __init__(self,
               img_shape, # tupple containing the dimentions of the image batch
               patch_size = 16, # patch dimentions
               num_transformer_layers = 12, # table 1 from layers in the paper
               embedding_dimention = 768, # hidden layer size "D" of the architecture
               mlp_size = 3072, # number of layers in multilayer perceptron
               num_heads = 12, # number of multi head self attention layers in encoder
               attn_dropout = 0.1, # droput of the attention layer
               mlp_dropout = 0.1, # the other dropout
               num_classes = 1000 # classes to classify (vary on the classification problem)
               ):
    super().__init__()

    # store hyperparameters
    self.batch_size,self.n_channels,self.height,self.width = img_shape
    self.patch_size = patch_size
    self.num_transformer_layers = num_transformer_layers
    self.embedding_dimention = embedding_dimention
    self.mlp_size = mlp_size
    self.num_heads = num_heads
    self.attn_dropout = attn_dropout
    self.mlp_dropout = mlp_dropout
    self.num_classes = num_classes
    
    # check whether the image is symetric
    assert self.height == self.width, "Image's height and width are different."
    # check whether the image's dimentions make sense with the proposed patch size and the image is simetrical
    assert self.height % self.patch_size == 0, f"The image of size {self.height} cannot be segmented into and {self.patch_size} patch size."

    self.num_patches = (self.height * self.width ) // self.patch_size ** 2
    
    # build architecture
    self.image_patcher = PatcherModule(
        img_shape = img_shape,
        patch_size = self.patch_size,
        embedding_dim = self.embedding_dimention
    )

    self.transformer_encoder = nn.Sequential(
        *[
            TransfomerEncoderModule(
                embedding_dim = self.embedding_dimention,
                num_heads = self.num_heads,
                mlp_size = self.mlp_size,
                mlp_dropout = self.mlp_dropout,
                attn_dropout=self.attn_dropout
            ) for _ in range(self.num_transformer_layers)
        ]
    )

    self.classifier = nn.Sequential(
        nn.LayerNorm(
            normalized_shape=self.embedding_dimention
        ),
        nn.Linear(
            in_features = self.embedding_dimention,
            out_features = self.num_classes
        )
    )
  def forward(self,x): # operator fusion optimized
    return self.classifier(self.transformer_encoder(self.image_patcher(x))[:,0])