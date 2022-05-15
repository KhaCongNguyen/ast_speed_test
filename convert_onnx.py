import torch.onnx
from models import ASTModel
import torch

# Function to Convert to ONNX
def Convert_ONNX():
    # set the model to inference mode
    audio_model.eval()

    # Create a dummy input tensor
    dummy_input = torch.rand([10, input_tdim, 128], dtype=torch.float16).cuda()
    
    # if run on cpu
    # dummy_input = torch.rand([10, input_tdim, 128])

    # Export the model
    torch.onnx.export(audio_model.module,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "best_audio_model.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Model's parameters
    label_dim = 50 # Number of classes 
    input_tdim = 512 # Number of features
    model_path = "best_audio_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=False, model_size='base384')

    # Load model
    sd = torch.load(model_path, map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # Conversion to ONNX
    Convert_ONNX()
