from models import ASTModel
import torch
import torchaudio
import os
from datetime import datetime

categories = {'dog': '0', 'rooster': '1', 'pig': '2', 'cow': '3', 'frog': '4', 'cat': '5', 'hen': '6', 'insects': '7',
              'sheep': '8', 'crow': '9', 'rain': '10', 'sea_waves': '11', 'crackling_fire': '12', 'crickets': '13',
              'chirping_birds': '14', 'water_drops': '15', 'wind': '16', 'pouring_water': '17', 'toilet_flush': '18',
              'thunderstorm': '19', 'crying_baby': '20', 'sneezing': '21', 'clapping': '22', 'breathing': '23',
              'coughing': '24', 'footsteps': '25', 'laughing': '26', 'brushing_teeth': '27', 'snoring': '28',
              'drinking_sipping': '29', 'door_wood_knock': '30', 'mouse_click': '31', 'keyboard_typing': '32',
              'door_wood_creaks': '33', 'can_opening': '34', 'washing_machine': '35', 'vacuum_cleaner': '36',
              'clock_alarm': '37', 'clock_tick': '38', 'glass_breaking': '39', 'helicopter': '40', 'chainsaw': '41',
              'siren': '42', 'car_horn': '43', 'engine': '44', 'train': '45', 'church_bells': '46', 'airplane': '47',
              'fireworks': '48', 'hand_saw': '49'}


def read_audio(file_path):
    """
    Read and process each audio file
    :param file_path: path of input file
    :return: tensor
    """
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform - waveform.mean()

    # create fbank feature form raw data
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

    # prepare for padding
    target_length = 512  # width of audio
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    # normalize
    fbank = (fbank + 6.6268077) / (5.358466 * 2)

    return fbank


if __name__ == "__main__":

    # environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Start testing speed of .pth AST model on %s g4dn.xlarge GPU" % torch.cuda.get_device_name(0))
    else:
        print("Start testing speed of .pth AST model on c5.xlarge CPU")

    # test data folder
    test_folder = "test_data"

    # prepare data
    list_data = list()
    for root, dir, files in os.walk(test_folder):
        for file in files:
            if not file.endswith("wav"):
                continue
            list_data.append({"name": file[:-4], "data": read_audio(os.path.join(root, file))})

    # concat tensors of input data
    input_data = torch.stack([file["data"] for file in list_data]).to(device)

    # model parameters
    label_dim = 50
    input_tdim = 512
    path = "best_audio_model.pth"

    # Start test speed, Start time
    start = datetime.now()

    # create and load model
    audio_model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=False, model_size='base384')
    sd = torch.load(path, map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # change to inference mode
    audio_model.eval()

    # predict
    predict = audio_model(input_data)

    # apply sigmoid to linear output
    predict_sigmoid = torch.sigmoid(predict)
    predict_cpu = predict_sigmoid.to('cpu').detach()
    cate_results = torch.argmax(predict_cpu, axis=1)

    # Get recognition result for each file
    for i, file in enumerate(list_data):
        print("The recognition result of file %s: %s" % (file["name"], list(categories.keys())[cate_results[i]]))

    # End time
    end = datetime.now()

    # Print total processing time
    print("Total execution time: " + str((end - start).total_seconds()) + "(seconds)")
