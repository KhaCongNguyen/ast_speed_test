import onnxruntime as nxrun
import numpy as np
import torchaudio
import torch
from datetime import datetime
import os

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
    :return: numpy array
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

    return fbank.numpy()


if __name__ == '__main__':
    # Start testing
    print("Start testing speed of Onnx AST model on C5 CPU")
    #print("Start testing speed of Onnx AST model on g4dn.xlarge GPU")

    # test data folder
    test_folder = "test_data"

    # prepare data
    list_data = list()
    for root, dir, files in os.walk(test_folder):
        for file in files:
            if not file.endswith("wav"):
                continue
            list_data.append({"name": file[:-4], "data": read_audio(os.path.join(root, file))})

    # convert to numpy
    np_audio_files = np.array([file["data"] for file in list_data]).astype(np.float16)

    # Start test speed, Start time
    start = datetime.now()

    # load onnx model
    sess = nxrun.InferenceSession("best_audio_model.onnx")

    # predict a batch of audio files
    results = sess.run(None, {"modelInput": np_audio_files})
    sigmoid_results = 1 / (1 + np.exp(-results[0]))
    cate_results = np.argmax(sigmoid_results, axis=1)

    # Get recognition result for each file
    for i, file in enumerate(list_data):
        print("The recognition result of file %s: %s" % (file["name"], list(categories.keys())[cate_results[i]]))

    # End time
    end = datetime.now()

    # Print total processing time
    print("Total execution time: " + str((end - start).total_seconds()) + "(seconds)")
