import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))

from tools.file_io import read_text

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

os.environ["OMP_NUM_THREADS"] = "4"

# CUDA Graph capture is unsafe with this real-time audio callback on the
# current Windows/PyTorch stack: capture can stall during startup and the UI
# only reports a generic "start error". Eager CUDA execution is reliable and
# still uses the GPU for all RVC inference.
os.environ["RVC_CUDA_GRAPH"] = "0"

realtime_config_path = os.path.join(now_dir, "configs", "config.json")

flag_vc = False


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


if __name__ == "__main__":
    import json
    import re
    import time
    import traceback

    import librosa
    from tools.torchgate import TorchGate
    import numpy as np
    import FreeSimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    from PySide6 import QtCore, QtGui, QtWidgets
    from ui_reference import MainWindow as ReferenceMainWindow, SliderRow as ReferenceSliderRow

    from configs.config import Config
    from infer import rtrvc as rvc_for_realtime
    from i18n.i18n import I18nAuto
    from tools.cuda_graph import clear_cuda_graph_cache, cuda_graph_enabled, run_cuda_graph

    i18n = I18nAuto()

    class GUIConfig:
        def __init__(self) :
            self.pth_path = ""
            self.index_path = ""
            self.pitch = 0
            self.formant=0.0
            self.sr_type = "sr_model"
            self.block_time = 0.25  # s
            self.threhold = -60
            self.crossfade_time = 0.05
            self.extra_time = 2.5
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.rms_mix_rate = 0.0
            self.index_rate = 0.0
            self.f0method = "rmvpe"
            self.sg_hostapi = ""
            self.wasapi_exclusive = False
            self.sg_input_device = ""
            self.sg_output_device = ""

    class GUI:
        def __init__(self) :
            self.gui_config = GUIConfig()
            self.config = Config()
            printt("RVC_CUDA_GRAPH=%s", os.environ.get("RVC_CUDA_GRAPH", "0"))
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                data = json.loads(read_text(realtime_config_path))
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                if data.get("f0method") not in ("pm", "rmvpe", "fcpe"):
                    data["f0method"] = "rmvpe"
                data["pm"] = data["f0method"] == "pm"
                data["rmvpe"] = data["f0method"] == "rmvpe"
                data["fcpe"] = data["f0method"] == "fcpe"
                if data["sg_hostapi"] in self.hostapis:
                    self.update_devices(hostapi_name=data["sg_hostapi"])
                    if (
                        data["sg_input_device"] not in self.input_devices
                        or data["sg_output_device"] not in self.output_devices
                    ):
                        self.update_devices()
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
                else:
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
            except:
                with open(realtime_config_path, "w", encoding="utf8") as j:
                    data = {
                        "pth_path": "",
                        "index_path": "",
                        "sg_hostapi": self.hostapis[0],
                        "sg_wasapi_exclusive": False,
                        "sg_input_device": self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ],
                        "sg_output_device": self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ],
                        "sr_type": "sr_model",
                        "threhold": -60,
                        "pitch": 0,
                        "formant": 0.0,
                        "index_rate": 0,
                        "rms_mix_rate": 0,
                        "block_time": 0.25,
                        "crossfade_length": 0.05,
                        "extra_time": 2.5,
                        "f0method": "rmvpe",
                    }
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    data["pm"] = data["f0method"] == "pm"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
                    data["fcpe"] = data["f0method"] == "fcpe"
            return data

        def launcher(self):
            data = self.load()
            sg.theme_add_new(
                "DeepVoiceDark",
                {
                    "BACKGROUND": "#1E1E1E",
                    "FRAME": "#252525",
                    "TEXT": "#F2F2F2",
                    "INPUT": "#2A2A2A",
                    "TEXT_INPUT": "#F2F2F2",
                    "SCROLL": "#2F6FEB",
                    "BUTTON": ("#FFFFFF", "#2F6FEB"),
                    "PROGRESS": ("#2F6FEB", "#2A2A2A"),
                    "BORDER": 1,
                    "SLIDER_DEPTH": 0,
                },
            )
            sg.theme("DeepVoiceDark")
            sg.set_options(font=("Segoe UI", 10), element_padding=(6, 5))
            layout = [
                [
                    sg.Frame(
                        title=i18n("加载模型"),
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("pth_path", ""),
                                    key="pth_path",
                                ),
                                sg.FileBrowse(
                                    i18n("选择.pth文件"),
                                    initial_folder=os.path.join(
                                        os.getcwd(), "assets/weights"
                                    ),
                                    file_types=((". pth"),),
                                ),
                            ],
                            [
                                sg.Input(
                                    default_text=data.get("index_path", ""),
                                    key="index_path",
                                ),
                                sg.FileBrowse(
                                    i18n("选择.index文件"),
                                    initial_folder=os.path.join(os.getcwd(), "logs"),
                                    file_types=((". index"),),
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("设备类型")),
                                sg.Combo(
                                    self.hostapis,
                                    key="sg_hostapi",
                                    default_value=data.get("sg_hostapi", ""),
                                    enable_events=True,
                                    size=(20, 1),
                                ),
                                sg.Checkbox(
                                    i18n("独占 WASAPI 设备"),
                                    key="sg_wasapi_exclusive",
                                    default=data.get("sg_wasapi_exclusive", False),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("输入设备")),
                                sg.Combo(
                                    self.input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Text(i18n("输出设备")),
                                sg.Combo(
                                    self.output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Button(i18n("重载设备列表"), key="reload_devices"),
                                sg.Radio(
                                    i18n("使用模型采样率"),
                                    "sr_type",
                                    key="sr_model",
                                    default=data.get("sr_model", True),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    i18n("使用设备采样率"),
                                    "sr_type",
                                    key="sr_device",
                                    default=data.get("sr_device", False),
                                    enable_events=True,
                                ),
                                sg.Text(i18n("采样率:")),
                                sg.Text("", key="sr_stream"),
                            ],
                        ],
                        title=i18n("音频设备"),
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("响应阈值")),
                                sg.Slider(
                                    range=(-60, 0),
                                    key="threhold",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("threhold", -60),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("音调设置")),
                                sg.Slider(
                                    range=(-16, 16),
                                    key="pitch",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("pitch", 0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("性别因子/声线粗细")),
                                sg.Slider(
                                    range=(-2, 2),
                                    key="formant",
                                    resolution=0.05,
                                    orientation="h",
                                    default_value=data.get("formant", 0.0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Index Rate")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="index_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("index_rate", 0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("响度因子")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="rms_mix_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("rms_mix_rate", 0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("音高算法")),
                                sg.Radio(
                                    "pm",
                                    "f0method",
                                    key="pm",
                                    default=data.get("pm", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "rmvpe",
                                    "f0method",
                                    key="rmvpe",
                                    default=data.get("rmvpe", True),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "fcpe",
                                    "f0method",
                                    key="fcpe",
                                    default=data.get("fcpe", False),
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title=i18n("常规设置"),
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("采样长度")),
                                sg.Slider(
                                    range=(0.02, 1.5),
                                    key="block_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("block_time", 0.25),
                                    enable_events=True,
                                ),
                            ],
                            # [
                            #     sg.Text("设备延迟"),
                            #     sg.Slider(
                            #         range=(0, 1),
                            #         key="device_latency",
                            #         resolution=0.001,
                            #         orientation="h",
                            #         default_value=data.get("device_latency", 0.1),
                            #         enable_events=True,
                            #     ),
                            # ],
                            [
                                sg.Text(i18n("淡入淡出长度")),
                                sg.Slider(
                                    range=(0.01, 0.15),
                                    key="crossfade_length",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("crossfade_length", 0.05),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("额外推理时长")),
                                sg.Slider(
                                    range=(0.05, 5.00),
                                    key="extra_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("extra_time", 2.5),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Checkbox(
                                    i18n("输入降噪"),
                                    key="I_noise_reduce",
                                    enable_events=True,
                                ),
                                sg.Checkbox(
                                    i18n("输出降噪"),
                                    key="O_noise_reduce",
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title=i18n("性能设置"),
                    ),
                ],
                [
                    sg.Button(i18n("开始音频转换"), key="start_vc"),
                    sg.Button(i18n("停止音频转换"), key="stop_vc"),
                    sg.Radio(
                        i18n("输入监听"),
                        "function",
                        key="im",
                        default=False,
                        enable_events=True,
                    ),
                    sg.Radio(
                        i18n("输出变声"),
                        "function",
                        key="vc",
                        default=True,
                        enable_events=True,
                    ),
                    sg.Text(i18n("算法延迟(ms):")),
                    sg.Text("0", key="delay_time"),
                    sg.Text(i18n("推理时间(ms):")),
                    sg.Text("0", key="infer_time"),
                ],
            ]
            self.window = sg.Window(
                "Deep Voice Live",
                layout=layout,
                finalize=True,
                resizable=True,
                background_color="#1E1E1E",
            )
            self.event_handler()

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis:
                        self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis)
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                    if (
                        self.gui_config.sg_input_device not in self.input_devices
                        and len(self.input_devices) > 0
                    ):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.gui_config.sg_input_device
                    )
                    if self.gui_config.sg_output_device not in self.output_devices:
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.gui_config.sg_output_device
                    )
                if event == "start_vc" and not flag_vc:
                    if self.set_values(values) == True:
                        printt(i18n("CUDA可用：%s"), torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_hostapi": values["sg_hostapi"],
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [
                                    values["sr_model"],
                                    values["sr_device"],
                                ].index(True)
                            ],
                            "threhold": values["threhold"],
                            "pitch": values["pitch"],
                            "rms_mix_rate": values["rms_mix_rate"],
                            "index_rate": values["index_rate"],
                            # "device_latency": values["device_latency"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "f0method": ["pm", "rmvpe", "fcpe"][
                                [values["pm"], values["rmvpe"], values["fcpe"]].index(True)
                            ],
                        }
                        with open(realtime_config_path, "w", encoding="utf8") as j:
                            json.dump(settings, j)
                        if self.stream is not None:
                            self.delay_time = (
                                self.stream.latency[-1]
                                + values["block_time"]
                                + values["crossfade_length"]
                                + 0.01
                            )
                        if values["I_noise_reduce"]:
                            self.delay_time += min(values["crossfade_length"], 0.04)
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                # Parameter hot update
                if event == "threhold":
                    self.gui_config.threhold = values["threhold"]
                elif event == "pitch":
                    self.gui_config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "formant":
                    self.gui_config.formant = values["formant"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_formant(values["formant"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in ["pm", "rmvpe", "fcpe"]:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    if self.stream is not None:
                        self.delay_time += (
                            1 if values["I_noise_reduce"] else -1
                        ) * min(values["crossfade_length"], 0.04)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event == "stop_vc" or event != "start_vc":
                    # Other parameters do not support hot update
                    self.stop_stream()

        def set_values(self, values):
            if len(values["pth_path"].strip()) == 0:
                sg.popup(i18n("请选择pth文件"))
                return False
            if len(values["index_path"].strip()) == 0:
                sg.popup(i18n("请选择index文件"))
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(values["pth_path"]):
                sg.popup(i18n("pth文件路径不可包含中文"))
                return False
            if pattern.findall(values["index_path"]):
                sg.popup(i18n("index文件路径不可包含中文"))
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            # self.device_latency = values["device_latency"]
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.formant = values["formant"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.f0method = ["pm", "rmvpe", "fcpe"][
                [values["pm"], values["rmvpe"], values["fcpe"]].index(True)
            ]
            return True

        def start_vc(self):
            # A microphone test uses the same RVC stream.  Starting another instance
            # while its callback is still active corrupts CUDA Graph capture, so always
            # release the old audio/GPU state before constructing a new converter.
            if self.stream is not None:
                self.stop_stream()
            for owner_name in ("rvc", "resampler", "resampler2"):
                owner = getattr(self, owner_name, None)
                if owner is not None:
                    clear_cuda_graph_cache(owner)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.config.device)
            torch.cuda.empty_cache()
            self.rvc = rvc_for_realtime.RVC(
                self.gui_config.pitch,
                self.gui_config.formant,
                self.gui_config.pth_path,
                self.gui_config.index_path,
                self.gui_config.index_rate,
                self.config,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.gui_config.samplerate = (
                self.rvc.tgt_sr
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.zc = self.gui_config.samplerate // 100
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.input_wav = torch.zeros(
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.input_wav_denoise = self.input_wav.clone()
            self.input_wav_res = torch.zeros(
                160 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
            self.sola_buffer = torch.zeros(
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
            )
            self.sola_den_kernel = torch.ones(
                1,
                1,
                self.sola_buffer_frame,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.nr_buffer = self.sola_buffer.clone()
            self.output_buffer = self.input_wav.clone()
            self.skip_head = self.extra_frame // self.zc
            self.return_length = (
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame
            ) // self.zc
            self.fade_in_window = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.rvc.tgt_sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.rvc.tgt_sr,
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            # Bundled torch.istft is not CUDA Graph-capturable, so TorchGate
            # stays eager while resampling and RVC inference still use graphs.
            self.tg = TorchGate(
                sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
            ).to(self.config.device)
            self.prewarm_cuda_graph()
            self.start_stream()

        def prewarm_cuda_graph(self):
            if not cuda_graph_enabled(self.config.device):
                return
            try:
                printt(i18n("正在预热CUDA Graph"))
                samples = self.input_wav_res.shape[0]
                phase = torch.arange(
                    samples, device=self.config.device, dtype=torch.float32
                )
                probe = 0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
                self.input_wav_res.copy_(probe)

                if self.gui_config.I_noise_reduce:
                    short = self.input_wav[
                        -self.sola_buffer_frame - self.block_frame :
                    ].unsqueeze(0)
                    self.tg(short, self.input_wav.unsqueeze(0))

                resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
                run_cuda_graph(
                    self.resampler,
                    "realtime-input-resample",
                    lambda audio: self.resampler(audio),
                    resample_input,
                )

                inferred = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    self.gui_config.f0method,
                )
                if self.resampler2 is not None:
                    inferred = run_cuda_graph(
                        self.resampler2,
                        "realtime-output-resample",
                        lambda audio: self.resampler2(audio),
                        inferred,
                    )
                if self.gui_config.O_noise_reduce:
                    self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
                torch.cuda.synchronize(self.config.device)
                printt(i18n("CUDA Graph预热完成"))
            except Exception:
                printt(traceback.format_exc())
            finally:
                self.input_wav.zero_()
                self.input_wav_denoise.zero_()
                self.input_wav_res.zero_()
                self.output_buffer.zero_()
                self.sola_buffer.zero_()
                self.nr_buffer.zero_()
                self.rvc.cache_pitch.zero_()
                self.rvc.cache_pitchf.zero_()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                if (
                    "WASAPI" in self.gui_config.sg_hostapi
                    and self.gui_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                def guarded_audio_callback(indata, outdata, frames, times, status):
                    try:
                        self.audio_callback(indata, outdata, frames, times, status)
                    except Exception:
                        # Never let an exception escape SoundDevice's callback:
                        # PortAudio otherwise stops the stream and leaves the UI
                        # looking active but completely silent.
                        outdata.fill(0)
                        if not getattr(self, "_callback_error_reported", False):
                            self._callback_error_reported = True
                            self._write_runtime_log(
                                "Audio callback failed\n" + traceback.format_exc()
                            )

                self._callback_error_reported = False
                self.stream = sd.Stream(
                    callback=guarded_audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config.samplerate,
                    channels=self.gui_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.stream.start()

        @staticmethod
        def _write_runtime_log(message):
            try:
                path = os.path.join(now_dir, "logs", "realtime_runtime.log")
                with open(path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        time.strftime("%Y-%m-%d %H:%M:%S")
                        + "  "
                        + message
                        + "\n"
                    )
            except OSError:
                pass

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None

        def audio_callback(
            self, indata, outdata, frames, times, status
        ):
            """
            音频处理
            """
            global flag_vc
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            if hasattr(self, "_set_input_level"):
                self._set_input_level(float(np.sqrt(np.mean(indata**2))))
            if hasattr(self, "_set_audio_samples"):
                self._set_audio_samples(indata[::max(1, len(indata) // 96)].copy())
            if self.gui_config.threhold > -60:
                indata = np.append(self.rms_buffer, indata)
                rms = librosa.feature.rms(
                    y=indata, frame_length=4 * self.zc, hop_length=self.zc
                )[:, 2:]
                self.rms_buffer[:] = indata[-4 * self.zc :]
                indata = indata[2 * self.zc - self.zc // 2 :]
                db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
                )
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
                indata = indata[self.zc // 2 :]
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
                self.config.device
            )
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            # input noise reduction and resampling
            if self.gui_config.I_noise_reduce:
                self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                    self.block_frame :
                ].clone()
                input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
                input_wav = self.tg(
                    input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
                ).squeeze(0)
                input_wav[: self.sola_buffer_frame] *= self.fade_in_window
                input_wav[: self.sola_buffer_frame] += (
                    self.nr_buffer * self.fade_out_window
                )
                self.input_wav_denoise[-self.block_frame :] = input_wav[
                    : self.block_frame
                ]
                self.nr_buffer[:] = input_wav[self.block_frame :]
                resample_input = self.input_wav_denoise[
                    -self.block_frame - 2 * self.zc :
                ]
                self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(
                    self.resampler,
                    "realtime-input-resample",
                    lambda audio: self.resampler(audio),
                    resample_input,
                )[160:]
            else:
                resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = run_cuda_graph(
                    self.resampler,
                    "realtime-input-resample",
                    lambda audio: self.resampler(audio),
                    resample_input,
                )[160:]
            # infer
            if self.function == "vc":
                infer_wav = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    self.gui_config.f0method,
                )
                if self.resampler2 is not None:
                    infer_wav = run_cuda_graph(
                        self.resampler2,
                        "realtime-output-resample",
                        lambda audio: self.resampler2(audio),
                        infer_wav,
                    )
            elif self.gui_config.I_noise_reduce:
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()
            # output noise reduction
            if self.gui_config.O_noise_reduce and self.function == "vc":
                self.output_buffer[: -self.block_frame] = self.output_buffer[
                    self.block_frame :
                ].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(
                    infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
                ).squeeze(0)
            # volume envelop mixing
            if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
                if self.gui_config.I_noise_reduce:
                    input_wav = self.input_wav_denoise[self.extra_frame :]
                else:
                    input_wav = self.input_wav[self.extra_frame :]
                rms1 = librosa.feature.rms(
                    y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                    frame_length=4 * self.zc,
                    hop_length=self.zc,
                )
                rms1 = torch.from_numpy(rms1).to(self.config.device)
                rms1 = F.interpolate(
                    rms1.unsqueeze(0),
                    size=infer_wav.shape[0] + 1,
                    mode="linear",
                    align_corners=True,
                )[0, 0, :-1]
                rms2 = librosa.feature.rms(
                    y=infer_wav[:].cpu().numpy(),
                    frame_length=4 * self.zc,
                    hop_length=self.zc,
                )
                rms2 = torch.from_numpy(rms2).to(self.config.device)
                rms2 = F.interpolate(
                    rms2.unsqueeze(0),
                    size=infer_wav.shape[0] + 1,
                    mode="linear",
                    align_corners=True,
                )[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(
                    rms1 / rms2, 1.0 - self.gui_config.rms_mix_rate
                )
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
            conv_input = infer_wav[
                None, None, : self.sola_buffer_frame + self.sola_search_frame
            ]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input**2,
                    self.sola_den_kernel,
                )
                + 1e-8
            )
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            printt(i18n("SOLA偏移：%d"), int(sola_offset))
            infer_wav = infer_wav[sola_offset:]
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
            self.sola_buffer[:] = infer_wav[
                self.block_frame : self.block_frame + self.sola_buffer_frame
            ]
            outdata[:] = (
                infer_wav[: self.block_frame]
                .repeat(self.gui_config.channels, 1)
                .t()
                .cpu()
                .numpy()
            )
            total_time = time.perf_counter() - start_time
            if flag_vc:
                infer_ms = int(total_time * 1000)
                if hasattr(self, "_set_infer_time"):
                    self._set_infer_time(infer_ms)
                elif hasattr(self, "window"):
                    self.window["infer_time"].update(infer_ms)
            printt(i18n("推理耗时：%.2f秒"), total_time)

        def update_devices(self, hostapi_name=None):
            """获取设备列表"""
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]

        def set_devices(self, input_device, output_device):
            """设置输出设备"""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt(i18n("输入设备：%s:%s"), str(sd.default.device[0]), input_device)
            printt(i18n("输出设备：%s:%s"), str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)

    class UiSignals(QtCore.QObject):
        infer_time = QtCore.Signal(int)
        input_level = QtCore.Signal(float)
        audio_samples = QtCore.Signal(object)


    class Waveform(QtWidgets.QWidget):
        """Small live visual driven by the real microphone level."""

        def __init__(self):
            super().__init__()
            self.level = 0.04
            self.phase = 0
            self.setMinimumHeight(92)

        def set_level(self, value):
            self.level = max(0.02, min(1.0, value * 12.0))
            self.phase += 1
            self.update()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect().adjusted(8, 8, -8, -8)
            painter.fillRect(rect, QtGui.QColor("#141924"))
            count = 48
            step = rect.width() / count
            for index in range(count):
                wave = 0.32 + 0.68 * abs(
                    np.sin((index + self.phase * 0.45) * 0.46)
                )
                height = max(5, rect.height() * (0.16 + self.level * wave * 0.84))
                x = rect.left() + index * step
                gradient = QtGui.QLinearGradient(x, rect.top(), x, rect.bottom())
                gradient.setColorAt(0, QtGui.QColor("#8B5CFF"))
                gradient.setColorAt(1, QtGui.QColor("#4B7BFF"))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(gradient)
                painter.drawRoundedRect(
                    QtCore.QRectF(x, rect.center().y() - height / 2, 3, height),
                    1.5,
                    1.5,
                )


    class HeroArtwork(QtWidgets.QWidget):
        """Decorative, resolution-independent artwork for the home card.

        It is drawn locally rather than loaded from a web image, so the program
        keeps the same look when it is started offline.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumSize(250, 145)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect().adjusted(2, 2, -2, -2)
            background = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
            background.setColorAt(0, QtGui.QColor("#15132D"))
            background.setColorAt(0.56, QtGui.QColor("#171838"))
            background.setColorAt(1, QtGui.QColor("#0B1020"))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, 9, 9)

            # Soft neon glow behind the profile.
            glow = QtGui.QRadialGradient(rect.right() - rect.width() * .29,
                                         rect.center().y(), rect.width() * .46)
            glow.setColorAt(0, QtGui.QColor(122, 76, 255, 105))
            glow.setColorAt(.54, QtGui.QColor(81, 63, 182, 38))
            glow.setColorAt(1, QtGui.QColor(10, 13, 30, 0))
            painter.setBrush(glow)
            painter.drawEllipse(rect)

            mid = rect.center().y()
            # Audio waveform that stays behind the profile.
            painter.setPen(QtGui.QPen(QtGui.QColor("#7E5BFF"), 2.2,
                                      QtCore.Qt.PenStyle.SolidLine,
                                      QtCore.Qt.PenCapStyle.RoundCap))
            left = rect.left() + rect.width() * .40
            right = rect.right() - 12
            step = max(7, (right - left) / 21)
            for i in range(22):
                x = left + i * step
                h = 7 + 23 * abs(np.sin(i * .68))
                painter.drawLine(QtCore.QPointF(x, mid - h), QtCore.QPointF(x, mid + h))

            # A clean headphone/profile silhouette.
            cx = rect.left() + rect.width() * .67
            cy = rect.top() + rect.height() * .50
            size = min(rect.width(), rect.height()) * .62
            outer = QtGui.QPen(QtGui.QColor("#9A70FF"), max(2.0, size * .036))
            outer.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(outer)
            painter.setBrush(QtGui.QColor(24, 22, 57, 190))
            painter.drawEllipse(QtCore.QRectF(cx - size*.30, cy - size*.43,
                                               size*.59, size*.86))
            painter.setBrush(QtGui.QColor("#1B1A41"))
            painter.drawRoundedRect(QtCore.QRectF(cx - size*.43, cy - size*.08,
                                                    size*.14, size*.30), 8, 8)
            painter.drawRoundedRect(QtCore.QRectF(cx + size*.27, cy - size*.08,
                                                    size*.14, size*.30), 8, 8)
            painter.drawArc(QtCore.QRectF(cx - size*.44, cy - size*.52, size*.88, size*.72), 28*16, 124*16)
            painter.drawLine(QtCore.QPointF(cx + size*.40, cy + size*.18),
                             QtCore.QPointF(cx + size*.57, cy + size*.38))
            painter.drawEllipse(QtCore.QRectF(cx + size*.53, cy + size*.34,
                                               size*.07, size*.07))


    class DropZone(QtWidgets.QLabel):
        files_dropped = QtCore.Signal(list)

        def __init__(self):
            super().__init__("Перетащите файлы модели сюда\nили выберите файлы вручную")
            self.setAcceptDrops(True)
            self.setObjectName("dropZone")
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        def dragEnterEvent(self, event):
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            if any(path.lower().endswith((".pth", ".index")) for path in paths):
                event.acceptProposedAction()
                self.setText("Отпустите файлы для загрузки")

        def dragLeaveEvent(self, event):
            self.setText("Перетащите файлы модели сюда\nили выберите файлы вручную")

        def dropEvent(self, event):
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            self.setText("Перетащите файлы модели сюда\nили выберите файлы вручную")
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


    class ToggleSwitch(QtWidgets.QAbstractButton):
        """A compact real on/off control used for live routing options."""

        def __init__(self, text="", parent=None):
            super().__init__(parent)
            self.setText(text)
            self.setCheckable(True)
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setMinimumHeight(24)
            self.setMinimumWidth(128)

        def sizeHint(self):
            return QtCore.QSize(max(128, self.fontMetrics().horizontalAdvance(self.text()) + 44), 24)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            toggle = QtCore.QRectF(2, (self.height() - 16) / 2, 28, 16)
            on = self.isChecked()
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#7457FF") if on else QtGui.QColor("#334158"))
            painter.drawRoundedRect(toggle, 8, 8)
            painter.setBrush(QtGui.QColor("#F7F8FF"))
            knob_x = toggle.right() - 14 if on else toggle.left() + 2
            painter.drawEllipse(QtCore.QRectF(knob_x, toggle.top() + 2, 12, 12))
            painter.setPen(QtGui.QColor("#EAF0FF"))
            painter.drawText(QtCore.QRectF(37, 0, self.width() - 37, self.height()),
                             QtCore.Qt.AlignmentFlag.AlignVCenter, self.text())


    class InputMeter(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.level = 0.0
            self.setFixedHeight(42)

        def set_level(self, value):
            self.level = max(0.0, min(1.0, value * 12.0))
            self.update()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect().adjusted(0, 8, 0, -8)
            count = 16
            gap = 3
            width = max(3, (rect.width() - gap * (count - 1)) / count)
            active = int(round(self.level * count))
            for i in range(count):
                shade = "#8A63FF" if i < active else "#29354B"
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(shade))
                painter.drawRoundedRect(QtCore.QRectF(rect.left() + i * (width + gap), rect.top(), width, rect.height()), 1.5, 1.5)


    class SectionIcon(QtWidgets.QWidget):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.setFixedSize(18, 18)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor("#8765FF"), 1.8)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            r = self.rect().adjusted(3, 2, -3, -2)
            if self.kind == "mic":
                painter.drawRoundedRect(QtCore.QRectF(6, 2, 6, 10), 3, 3)
                painter.drawArc(QtCore.QRectF(3.5, 6, 11, 9), 0, -180 * 16)
                painter.drawLine(9, 15, 9, 17)
                painter.drawLine(6, 17, 12, 17)
            elif self.kind == "wave":
                for x, h in ((3, 5), (6, 11), (9, 15), (12, 9), (15, 5)):
                    painter.drawLine(x, 9 - h / 2, x, 9 + h / 2)
            elif self.kind == "tune":
                for y, x in ((4, 8), (9, 12), (14, 6)):
                    painter.drawLine(3, y, 16, y)
                    painter.drawEllipse(QtCore.QRectF(x - 1.5, y - 1.5, 3, 3))
            elif self.kind == "clock":
                painter.drawEllipse(r)
                painter.drawLine(9, 5, 9, 10)
                painter.drawLine(9, 10, 12, 12)
            else:
                painter.drawRoundedRect(r, 2, 2)
                painter.drawLine(6, 6, 12, 6)
                painter.drawLine(6, 10, 12, 10)


    class NavButton(QtWidgets.QAbstractButton):
        def __init__(self, text, kind):
            super().__init__()
            self.setText(text)
            self.kind = kind
            self.setCheckable(True)
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect().adjusted(1, 1, -1, -1)
            if self.isChecked():
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor("#6C54E9"))
                painter.drawRoundedRect(rect, 7, 7)
            elif self.underMouse():
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor("#1D2638"))
                painter.drawRoundedRect(rect, 7, 7)
            color = QtGui.QColor("#FFFFFF") if self.isChecked() else QtGui.QColor("#B4C0D8")
            pen = QtGui.QPen(color, 1.5)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            cx = 22
            if self.kind == "home":
                painter.drawPolyline([QtCore.QPointF(cx - 8, 20), QtCore.QPointF(cx, 13), QtCore.QPointF(cx + 8, 20)])
                painter.drawRect(QtCore.QRectF(cx - 5, 19, 10, 9))
            elif self.kind == "profile":
                painter.drawEllipse(QtCore.QRectF(cx - 4, 14, 8, 8))
                painter.drawArc(QtCore.QRectF(cx - 8, 21, 16, 10), 25 * 16, 130 * 16)
            elif self.kind == "hotkey":
                painter.drawRoundedRect(QtCore.QRectF(cx - 9, 16, 18, 12), 3, 3)
                painter.drawLine(cx - 5, 20, cx + 5, 20)
                painter.drawLine(cx - 5, 24, cx + 1, 24)
            elif self.kind == "settings":
                painter.drawEllipse(QtCore.QRectF(cx - 6, 15, 12, 12))
                painter.drawEllipse(QtCore.QRectF(cx - 2, 19, 4, 4))
            elif self.kind == "logs":
                painter.drawRoundedRect(QtCore.QRectF(cx - 7, 14, 14, 16), 2, 2)
                for y in (18, 22, 26): painter.drawLine(cx - 4, y, cx + 4, y)
            painter.setPen(color)
            text_rect = QtCore.QRectF(43, 0, self.width() - 48, self.height())
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.TextFlag.TextWordWrap, self.text())


    class FeatureBadge(QtWidgets.QWidget):
        def __init__(self, title, kind):
            super().__init__()
            self.title = title
            self.kind = kind
            self.setMinimumWidth(56)
            self.setFixedHeight(61)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            circle = QtCore.QRectF((self.width() - 32) / 2, 1, 32, 32)
            painter.setPen(QtGui.QPen(QtGui.QColor("#5C48BE"), 1))
            painter.setBrush(QtGui.QColor("#1B1941"))
            painter.drawEllipse(circle)
            painter.setPen(QtGui.QPen(QtGui.QColor("#B39AFF"), 1.7))
            c = circle.center()
            if self.kind == 0:
                painter.drawEllipse(QtCore.QRectF(c.x()-5, c.y()-6, 10, 12))
                painter.drawLine(c.x(), c.y()+6, c.x(), c.y()+10)
            elif self.kind == 1:
                painter.drawEllipse(QtCore.QRectF(c.x()-5, c.y()-5, 10, 10))
                painter.drawLine(c.x(), c.y()-3, c.x(), c.y()+2)
                painter.drawLine(c.x(), c.y()+2, c.x()+4, c.y()+5)
            elif self.kind == 2:
                painter.drawRect(QtCore.QRectF(c.x()-6, c.y()-5, 12, 10))
                painter.drawLine(c.x()-3, c.y(), c.x(), c.y()+3)
                painter.drawLine(c.x(), c.y()+3, c.x()+4, c.y()-3)
            elif self.kind == 3:
                painter.drawArc(QtCore.QRectF(c.x()-7, c.y()-7, 14, 14), 25*16, 130*16)
                painter.drawLine(c.x()+5, c.y()+5, c.x()+8, c.y()+8)
            else:
                painter.drawEllipse(QtCore.QRectF(c.x()-6, c.y()-6, 12, 12))
                painter.drawEllipse(QtCore.QRectF(c.x()-2, c.y()-2, 4, 4))
            painter.setPen(QtGui.QColor("#E2DCFF"))
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            painter.drawText(QtCore.QRectF(0, 37, self.width(), 24), QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.TextFlag.TextWordWrap, self.title)


    class ModernGUI(GUI):
        """Qt front end; it reuses the tested RVC audio and CUDA methods above."""

        def _card(self, title, icon="model"):
            frame = QtWidgets.QFrame()
            frame.setObjectName("card")
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(10)
            if title:
                heading_row = QtWidgets.QHBoxLayout()
                heading_row.setSpacing(6)
                heading_row.addWidget(SectionIcon(icon))
                heading = QtWidgets.QLabel(title)
                heading.setObjectName("cardTitle")
                heading_row.addWidget(heading)
                heading_row.addStretch(1)
                layout.addLayout(heading_row)
            return frame, layout

        def _combo(self, values, current):
            combo = QtWidgets.QComboBox()
            combo.addItems(values)
            if current in values:
                combo.setCurrentText(current)
            return combo

        def _file_row(self, label, value, suffix, folder):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            field = QtWidgets.QLineEdit(value)
            row.addWidget(field, 1)
            browse = QtWidgets.QPushButton("Выбрать файл " + suffix)
            browse.setObjectName("outlineButton")

            def choose_file():
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.ui, "Выбрать файл", folder, "*" + suffix
                )
                if path:
                    field.setText(path)

            browse.clicked.connect(choose_file)
            row.addWidget(browse)
            return row, field

        def _slider_row(self, title, key, low, high, value, decimals=2):
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(title)
            label.setMinimumWidth(168)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setSingleStep(1 if decimals == 0 else 0.01)
            spin.setFixedWidth(78)

            def to_value(position):
                return low + (high - low) * position / 1000

            def to_position(number):
                return int(round((number - low) * 1000 / (high - low)))

            initial = max(low, min(high, float(value)))
            slider.setValue(to_position(initial))
            spin.setValue(initial)

            def slider_changed(position):
                number = to_value(position)
                spin.blockSignals(True)
                spin.setValue(number)
                spin.blockSignals(False)
                self._hot_update(key, number)

            def spin_changed(number):
                slider.blockSignals(True)
                slider.setValue(to_position(number))
                slider.blockSignals(False)
                self._hot_update(key, number)

            slider.valueChanged.connect(slider_changed)
            spin.valueChanged.connect(spin_changed)
            row.addWidget(label)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            self.controls[key] = spin
            return row

        def _hot_update(self, key, value):
            setattr(self.gui_config, key if key != "crossfade_length" else "crossfade_time", value)
            if not hasattr(self, "rvc"):
                return
            if key == "pitch":
                self.rvc.change_key(value)
            elif key == "formant":
                self.rvc.change_formant(value)
            elif key == "index_rate":
                self.rvc.change_index_rate(value)

        def launcher(self):
            data = self.load()
            self.controls = {}
            self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            self.qt_app.setApplicationName("Deep Voice Live")
            self.qt_app.setStyleSheet(
                """
                * { font-family: 'Segoe UI'; color: #EAF0FF; font-size: 13px; }
                QMainWindow { background: #090D15; }
                QWidget#central, QWidget#workspace { background: #090D15; }
                QLabel, QRadioButton, QCheckBox { background: transparent; }
                QFrame#topBar { background: #0C111D; border-bottom: 1px solid #202A3E; }
                QFrame#sidebar { background: #101624; border: 1px solid #202B42; border-radius: 10px; }
                QFrame#card { background: #111827; border: 1px solid #273249; border-radius: 10px; }
                QFrame#hero { background: #111429; border: 1px solid #684EDE; border-radius: 10px; }
                QLabel#cardTitle { color: #F4F6FF; font-size: 15px; font-weight: 700; }
                QLabel#heroTitle { color: #D8D2FF; font-size: 24px; font-weight: 700; }
                QLabel#heroSyrex { color: #8A62FF; font-size: 22px; font-weight: 700; }
                QPushButton#navActive { background: #6C54E9; border: 1px solid #917FFF; color: #FFFFFF; border-radius: 7px; font-weight: 700; padding: 7px; }
                QPushButton#navItem { background: transparent; border: 1px solid transparent; color: #A9B4C9; border-radius: 7px; padding: 7px; }
                QPushButton#navItem:hover { background:#1D2638; border-color:#303E58; color:#FFFFFF; }
                QLabel#dropZone { color: #C9D3E8; border: 1px dashed #4B5976; border-radius: 8px; padding: 18px; background: #10182A; }
                QLineEdit, QComboBox, QDoubleSpinBox { background: #171F2D; border: 1px solid #34435C; border-radius: 7px; min-height: 28px; padding: 0 9px; }
                QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #7A5CFF; }
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; height: 0; border: none; image: none; }
                QComboBox::drop-down { border: none; width: 22px; }
                QPushButton { background: #242E40; border: 1px solid #34435C; border-radius: 8px; min-height: 30px; padding: 0 12px; font-weight: 600; }
                QPushButton:hover { border-color: #8067FF; background: #303D55; }
                QPushButton#liveButton { background: #7057F5; border-color: #8F7CFF; min-height: 40px; font-size: 14px; }
                QPushButton#liveButton:hover { background: #8068FF; }
                QPushButton#outlineButton { color: #B7A9FF; border-color: #725BEB; background: #151426; }
                QPushButton#windowButton { border: none; background: transparent; color: #AAB5CC; min-width: 30px; max-width: 30px; padding: 0; font-size: 15px; }
                QPushButton#windowButton:hover { background: #2A3348; color: #FFFFFF; }
                QPushButton#closeButton:hover { background: #C53A52; color: #FFFFFF; }
                QListWidget { background:#0F1522; border:1px solid #273249; border-radius:7px; outline: none; }
                QListWidget::item { padding:6px 8px; border-radius:5px; }
                QListWidget::item:hover { background:#202A3D; }
                QSlider::groove:horizontal { height: 5px; background: #344158; border-radius: 2px; }
                QSlider::sub-page:horizontal { background: #7D5CFF; border-radius: 2px; }
                QSlider::handle:horizontal { background: #F1F3FF; border: 1px solid #C7CBDA; width: 13px; height: 13px; margin: -5px 0; border-radius: 7px; }
                QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #50607D; border-radius: 3px; background: #0E1626; }
                QCheckBox::indicator:hover { border-color: #9278FF; }
                QCheckBox::indicator:checked { background: #7057F5; border: 1px solid #A89AFF; image: none; }
                QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid #50607D; border-radius: 7px; background: #0E1626; }
                QRadioButton::indicator:checked { background: #7057F5; border: 3px solid #E9E6FF; }
                """
            )
            self.ui = QtWidgets.QMainWindow()
            self.ui.setWindowTitle("Deep Voice Live")
            self.ui.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Window)
            self.ui.setMinimumSize(1080, 680)
            self.ui.resize(1600, 900)
            central = QtWidgets.QWidget()
            central.setObjectName("central")
            root = QtWidgets.QVBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 10)
            root.setSpacing(10)

            topbar = QtWidgets.QFrame()
            topbar.setObjectName("topBar")
            topbar.setFixedHeight(64)
            header = QtWidgets.QHBoxLayout(topbar)
            header.setContentsMargins(22, 8, 12, 8)
            icon = QtWidgets.QLabel("▮▮")
            icon.setStyleSheet("color:#A675FF; font-size:28px; font-weight:700; letter-spacing:3px;")
            header.addWidget(icon)
            names = QtWidgets.QVBoxLayout()
            name = QtWidgets.QLabel("Deep Voice Live")
            name.setStyleSheet("font-size:20px; font-weight:700;")
            names.addWidget(name)
            subtitle = QtWidgets.QLabel("Real-time AI Voice Changer")
            subtitle.setStyleSheet("color:#9FAAC1;")
            names.addWidget(subtitle)
            header.addLayout(names)
            header.addStretch(1)
            ready = QtWidgets.QLabel("●  Готов к работе")
            ready.setStyleSheet("color:#82F4B2;")
            header.addWidget(ready)
            runtime = QtWidgets.QLabel("CPU 5%     RAM 412 MB     ▾")
            runtime.setStyleSheet("color:#AAB5CC; background:#121A28; border:1px solid #293650; border-radius:7px; padding:5px 9px;")
            header.addWidget(runtime)
            for symbol, callback, close in (("—", self.ui.showMinimized, False), ("□", self.ui.showMaximized, False), ("×", self.ui.close, True)):
                button = QtWidgets.QPushButton(symbol)
                button.setObjectName("closeButton" if close else "windowButton")
                button.clicked.connect(callback)
                header.addWidget(button)
            root.addWidget(topbar)

            sidebar = QtWidgets.QFrame()
            sidebar.setObjectName("sidebar")
            sidebar.setFixedWidth(165)
            sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(8, 12, 8, 12)
            sidebar_layout.setSpacing(5)
            self.nav_buttons = []
            self.nav_page_map = []
            nav_specs = (("Главная", "home", 0), ("Загрузка модели", "model", 0), ("Профили", "profile", 1),
                         ("Горячие клавиши", "hotkey", 2), ("Микшер", "tune", 3), ("Эффекты", "wave", 3),
                         ("Логи", "logs", 4), ("Настройки", "settings", 3))
            for index, (text, kind, page) in enumerate(nav_specs):
                item = NavButton(text, kind)
                item.setChecked(index == 0)
                item.setObjectName("navButton")
                item.setMinimumHeight(45)
                item.clicked.connect(lambda checked=False, page=page, nav=index: self._switch_page(page, nav))
                self.nav_buttons.append(item)
                self.nav_page_map.append(page)
                sidebar_layout.addWidget(item)
            sidebar_layout.addStretch(1)
            about = QtWidgets.QPushButton("О программе")
            about.setObjectName("navItem")
            about.setMinimumHeight(45)
            about.clicked.connect(lambda: self._switch_page(5))
            sidebar_layout.addWidget(about)

            body = QtWidgets.QHBoxLayout()
            body.setSpacing(12)
            body.setContentsMargins(16, 0, 16, 0)
            body.addWidget(sidebar)

            content = QtWidgets.QGridLayout()
            content.setHorizontalSpacing(12)
            content.setVerticalSpacing(12)
            model_card, model_layout = self._card("Загрузка модели", "model")
            model_card.setMinimumHeight(345)
            model_split = QtWidgets.QHBoxLayout()
            model_split.setSpacing(12)
            model_left = QtWidgets.QVBoxLayout()
            drop_zone = DropZone()
            drop_zone.files_dropped.connect(self._load_dropped_files)
            self.drop_zone = drop_zone
            model_left.addWidget(drop_zone)
            row, self.pth_field = self._file_row("Файл модели (.pth)", data.get("pth_path", ""), ".pth", os.path.join(now_dir, "assets", "weights"))
            model_left.addLayout(row)
            row, self.index_field = self._file_row("Файл индекса (.index)", data.get("index_path", ""), ".index", os.path.join(now_dir, "assets", "indices"))
            model_left.addLayout(row)
            model_split.addLayout(model_left, 1)
            recent_wrap = QtWidgets.QVBoxLayout()
            recent = QtWidgets.QLabel("Последние модели")
            recent.setStyleSheet("color:#9FAAC1; font-weight:600;")
            recent_wrap.addWidget(recent)
            recent_models = QtWidgets.QListWidget()
            recent_models.setMinimumWidth(165)
            recent_models.setMaximumHeight(142)
            weight_dir = os.path.join(now_dir, "assets", "weights")
            for model_name in sorted(os.listdir(weight_dir), reverse=True)[:3]:
                if model_name.lower().endswith(".pth"):
                    recent_models.addItem(model_name)
            recent_models.itemClicked.connect(lambda item: self.pth_field.setText(os.path.join(weight_dir, item.text())))
            recent_wrap.addWidget(recent_models, 1)
            model_split.addLayout(recent_wrap)
            model_layout.addLayout(model_split)
            content.addWidget(model_card, 0, 0)

            info_card, info_layout = self._card("", "model")
            info_card.setObjectName("hero")
            info_card.setMinimumHeight(345)
            # The hero follows the supplied reference: copy and controls on the
            # left, a local neon profile illustration on the right.
            hero_top = QtWidgets.QHBoxLayout()
            hero_copy = QtWidgets.QVBoxLayout()
            hero_name = QtWidgets.QLabel("Deep Voice Live")
            hero_name.setObjectName("heroTitle")
            hero_copy.addWidget(hero_name)
            hero_syrex = QtWidgets.QLabel("by Syrex")
            hero_syrex.setObjectName("heroSyrex")
            hero_copy.addWidget(hero_syrex)
            description = QtWidgets.QLabel("Профессиональный RVC\nголосовой конвертер\nв реальном времени")
            description.setStyleSheet("font-size:13px; color:#B5C0D8;")
            hero_copy.addWidget(description)
            quote = QtWidgets.QLabel("«Ваш голос. Больше возможностей.»")
            quote.setStyleSheet("color:#AFB8D1; font-style:italic;")
            hero_copy.addWidget(quote)
            hero_copy.addStretch(1)
            hero_top.addLayout(hero_copy, 3)
            hero_image = QtWidgets.QLabel()
            hero_image.setMinimumWidth(255)
            hero_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            hero_image.setPixmap(QtGui.QPixmap(os.path.join(now_dir, "assets", "ui", "hero-syrex.png")).scaled(330, 250, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding, QtCore.Qt.TransformationMode.SmoothTransformation))
            hero_top.addWidget(hero_image, 4)
            version = QtWidgets.QLabel("v1.0.0")
            version.setStyleSheet("color:#B8A9FF; border:1px solid #6C54E9; border-radius:8px; padding:4px 9px;")
            version.setParent(info_card)
            info_layout.addLayout(hero_top, 1)
            capabilities = QtWidgets.QHBoxLayout()
            capabilities.setSpacing(5)
            for badge_name, badge_kind in (("RVC / AI", 0), ("Real-time", 1), ("Высокое\nкачество", 2), ("Низкая\nзадержка", 3), ("Простая\nнастройка", 4)):
                capabilities.addWidget(FeatureBadge(badge_name, badge_kind))
            capabilities.addStretch(1)
            link_buttons = QtWidgets.QVBoxLayout()
            for text in ("GitHub", "Discord", "Поддержать", "Документация"):
                link = QtWidgets.QPushButton(text)
                link.setObjectName("outlineButton")
                link.setMinimumHeight(22)
                link.setMaximumHeight(24)
                link_buttons.addWidget(link)
            capability_block = QtWidgets.QHBoxLayout()
            capability_block.addLayout(capabilities, 1)
            capability_block.addLayout(link_buttons)
            info_layout.addLayout(capability_block)
            quote_box = QtWidgets.QLabel("«Голос — это не просто звук.\nЭто мой инструмент.»")
            quote_box.setStyleSheet("color:#C6B9FF; font-style:italic; background:#191437; border:1px solid #4B347C; border-radius:7px; padding:7px;")
            info_layout.addWidget(quote_box)
            signature = QtWidgets.QLabel("— by Syrex")
            signature.setStyleSheet("color:#B69AFF; font-style:italic;")
            info_layout.addWidget(signature, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            version.move(0, 0)
            def position_version(event):
                version.move(info_card.width() - version.width() - 16, 16)
            info_card.resizeEvent = position_version
            content.addWidget(info_card, 0, 1)

            device_card, device_layout = self._card("Аудиоустройство", "mic")
            device_split = QtWidgets.QHBoxLayout()
            device_controls = QtWidgets.QVBoxLayout()
            form = QtWidgets.QFormLayout()
            self.host_combo = self._combo(self.hostapis, data.get("sg_hostapi", ""))
            form.addRow("Тип устройства", self.host_combo)
            self.input_combo = self._combo(self.input_devices, data.get("sg_input_device", ""))
            form.addRow("Входное устройство", self.input_combo)
            self.output_combo = self._combo(self.output_devices, data.get("sg_output_device", ""))
            form.addRow("Выходное устройство", self.output_combo)
            device_controls.addLayout(form)
            device_options = QtWidgets.QHBoxLayout()
            self.wasapi_box = QtWidgets.QCheckBox("Эксклюзивный WASAPI")
            self.wasapi_box.setChecked(data.get("sg_wasapi_exclusive", False))
            device_options.addWidget(self.wasapi_box)
            refresh = QtWidgets.QPushButton("⟳")
            refresh.setFixedWidth(36)
            refresh.clicked.connect(self._refresh_devices)
            device_options.addWidget(refresh)
            self.rate_model = QtWidgets.QRadioButton("Частота модели")
            self.rate_device = QtWidgets.QRadioButton("Частота устройства")
            self.rate_model.setChecked(data.get("sr_model", True))
            self.rate_device.setChecked(data.get("sr_device", False))
            device_options.addWidget(self.rate_model)
            device_options.addWidget(self.rate_device)
            device_controls.addLayout(device_options)
            device_split.addLayout(device_controls, 1)
            test_panel = QtWidgets.QFrame()
            test_panel.setObjectName("card")
            test_panel.setMinimumWidth(145)
            test_layout = QtWidgets.QVBoxLayout(test_panel)
            test_layout.setContentsMargins(10, 8, 10, 8)
            test_layout.setSpacing(4)
            test_caption = QtWidgets.QLabel("Тест микрофона")
            test_caption.setStyleSheet("color:#D7D0FF; font-weight:600;")
            test_layout.addWidget(test_caption)
            self.input_meter = InputMeter()
            test_layout.addWidget(self.input_meter)
            self.mic_level_label = QtWidgets.QLabel("— dB")
            self.mic_level_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.mic_level_label.setStyleSheet("color:#AAB5CC; font-size:11px;")
            test_layout.addWidget(self.mic_level_label)
            mic_test = QtWidgets.QPushButton("▶  Проверить звук")
            mic_test.setObjectName("outlineButton")
            mic_test.clicked.connect(self._test_microphone)
            test_layout.addWidget(mic_test)
            device_split.addWidget(test_panel)
            device_layout.addLayout(device_split)
            content.addWidget(device_card, 1, 0)

            waveform_card, waveform_layout = self._card("Входной сигнал", "wave")
            self.waveform = Waveform()
            waveform_layout.addWidget(self.waveform)
            self.db_label = QtWidgets.QLabel("— dB")
            self.db_label.setStyleSheet("color:#AEB8CC;")
            waveform_layout.addWidget(self.db_label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            content.addWidget(waveform_card, 1, 1)

            voice_card, voice_layout = self._card("Основные настройки", "tune")
            voice_layout.addLayout(self._slider_row("Порог ответа", "threhold", -60, 0, data.get("threhold", -60), 0))
            voice_layout.addLayout(self._slider_row("Высота голоса", "pitch", -12, 12, data.get("pitch", 0), 0))
            voice_layout.addLayout(self._slider_row("Тембр / толщина", "formant", -1, 1, data.get("formant", 0), 2))
            voice_layout.addLayout(self._slider_row("Темп индекса", "index_rate", 0, 1, data.get("index_rate", 0.7), 2))
            voice_layout.addLayout(self._slider_row("Громкость", "rms_mix_rate", 0, 1, data.get("rms_mix_rate", 1), 2))
            pitch_row = QtWidgets.QHBoxLayout()
            pitch_row.addWidget(QtWidgets.QLabel("Алгоритм высоты тона"))
            self.pm_radio = QtWidgets.QRadioButton("PM")
            self.rmvpe_radio = QtWidgets.QRadioButton("RMVPE")
            self.fcpe_radio = QtWidgets.QRadioButton("FCPE")
            {"pm": self.pm_radio, "rmvpe": self.rmvpe_radio, "fcpe": self.fcpe_radio}.get(data.get("f0method", "rmvpe"), self.rmvpe_radio).setChecked(True)
            for button in (self.pm_radio, self.rmvpe_radio, self.fcpe_radio):
                pitch_row.addWidget(button)
            voice_layout.addLayout(pitch_row)
            content.addWidget(voice_card, 2, 0)

            perf_card, perf_layout = self._card("Настройки быстроты", "clock")
            perf_layout.addLayout(self._slider_row("Длина сэмпла", "block_time", 0.01, 1, data.get("block_time", 0.13), 2))
            perf_layout.addLayout(self._slider_row("Длина затухания", "crossfade_length", 0.01, 1, data.get("crossfade_length", 0.08), 2))
            perf_layout.addLayout(self._slider_row("Доп. время обработки", "extra_time", 0, 10, data.get("extra_time", 2), 2))
            boxes = QtWidgets.QHBoxLayout()
            self.in_noise = QtWidgets.QCheckBox("Уменьшение входного шума")
            self.out_noise = QtWidgets.QCheckBox("Уменьшение выходного шума")
            boxes.addWidget(self.in_noise)
            boxes.addWidget(self.out_noise)
            perf_layout.addLayout(boxes)
            note = QtWidgets.QLabel("ⓘ  Шумоподавление включайте только при заметном шуме микрофона.")
            note.setWordWrap(True)
            note.setStyleSheet("color:#AAB5CC; background:#182238; padding:9px; border-radius:7px;")
            perf_layout.addWidget(note)
            content.addWidget(perf_card, 2, 1)
            content.setColumnStretch(0, 1)
            content.setColumnStretch(1, 1)
            root.addLayout(body, 1)

            bottom = QtWidgets.QHBoxLayout()
            self.live_button = QtWidgets.QPushButton("▶  Начать конвертацию аудио")
            self.live_button.setObjectName("liveButton")
            self.live_button.clicked.connect(self._start_live)
            stop = QtWidgets.QPushButton("■  Остановить")
            stop.clicked.connect(self._stop_live)
            bottom.addWidget(self.live_button)
            bottom.addWidget(stop)
            self.monitor_radio = ToggleSwitch("Мониторинг входа")
            self.convert_radio = ToggleSwitch("Преобразование выхода")
            self.convert_radio.setChecked(True)
            self.monitor_radio.toggled.connect(self._set_function)
            self.convert_radio.toggled.connect(self._set_function)
            bottom.addWidget(self.monitor_radio)
            bottom.addWidget(self.convert_radio)
            bottom.addStretch(1)
            bottom.addWidget(QtWidgets.QLabel("Задержка (мс)"))
            self.delay_label = QtWidgets.QLabel("—")
            self.delay_label.setObjectName("metric")
            bottom.addWidget(self.delay_label)
            bottom.addWidget(QtWidgets.QLabel("Обработка (мс)"))
            self.infer_label = QtWidgets.QLabel("—")
            self.infer_label.setObjectName("metric")
            bottom.addWidget(self.infer_label)
            credit = QtWidgets.QLabel("Deep Voice Live by Syrex")
            credit.setStyleSheet("color:#9589D6; font-size:11px;")
            bottom.addWidget(credit)
            root.addLayout(bottom)

            self.signals = UiSignals()
            self._logs = ["Deep Voice Live запущен."]
            self.signals.infer_time.connect(lambda value: self.infer_label.setText(str(value)))
            self.signals.input_level.connect(self._update_signal)
            self.host_combo.currentTextChanged.connect(self._refresh_devices)
            home_page = QtWidgets.QWidget()
            home_page.setObjectName("workspace")
            home_page.setLayout(content)
            self.pages = QtWidgets.QStackedWidget()
            self.pages.addWidget(home_page)
            self.pages.addWidget(self._build_profiles_page())
            self.pages.addWidget(self._build_hotkeys_page())
            self.pages.addWidget(self._build_settings_page())
            self.pages.addWidget(self._build_logs_page())
            self.pages.addWidget(self._build_about_page())
            body.addWidget(self.pages, 1)
            self._configure_shortcuts()
            self.ui.setCentralWidget(central)
            self.ui.destroyed.connect(lambda: self.stop_stream())
            self.qt_app.aboutToQuit.connect(lambda: self._save_settings())
            self.ui.show()
            self.qt_app.exec()

        def _page(self, title, subtitle):
            page = QtWidgets.QWidget()
            page.setObjectName("workspace")
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            card, card_layout = self._card(title)
            description = QtWidgets.QLabel(subtitle)
            description.setStyleSheet("color:#AAB5CC;")
            card_layout.addWidget(description)
            layout.addWidget(card)
            return page, layout

        def _build_profiles_page(self):
            page, layout = self._page("Профили", "Сохраняйте наборы параметров голоса и быстро возвращайтесь к ним.")
            card = layout.itemAt(0).widget()
            card_layout = card.layout()
            row = QtWidgets.QHBoxLayout()
            self.profile_name = QtWidgets.QLineEdit()
            self.profile_name.setPlaceholderText("Название нового профиля")
            save = QtWidgets.QPushButton("Сохранить текущий профиль")
            save.setObjectName("outlineButton")
            save.clicked.connect(self._save_profile)
            row.addWidget(self.profile_name, 1)
            row.addWidget(save)
            card_layout.addLayout(row)
            self.profile_list = QtWidgets.QListWidget()
            self.profile_list.setMinimumHeight(200)
            self.profile_list.itemDoubleClicked.connect(self._load_profile)
            card_layout.addWidget(self.profile_list)
            actions = QtWidgets.QHBoxLayout()
            load = QtWidgets.QPushButton("Загрузить выбранный")
            load.clicked.connect(lambda: self._load_profile(self.profile_list.currentItem()))
            remove = QtWidgets.QPushButton("Удалить выбранный")
            remove.clicked.connect(self._delete_profile)
            actions.addWidget(load)
            actions.addWidget(remove)
            actions.addStretch(1)
            card_layout.addLayout(actions)
            self._refresh_profiles()
            layout.addStretch(1)
            return page

        def _build_hotkeys_page(self):
            page, layout = self._page("Горячие клавиши", "Клавиши работают, пока окно Deep Voice Live находится в фокусе.")
            card = layout.itemAt(0).widget()
            card_layout = card.layout()
            self.start_hotkey = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence("Ctrl+Shift+S"))
            self.stop_hotkey = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence("Ctrl+Shift+X"))
            form = QtWidgets.QFormLayout()
            form.addRow("Начать конвертацию", self.start_hotkey)
            form.addRow("Остановить конвертацию", self.stop_hotkey)
            card_layout.addLayout(form)
            apply = QtWidgets.QPushButton("Применить горячие клавиши")
            apply.setObjectName("outlineButton")
            apply.clicked.connect(self._configure_shortcuts)
            card_layout.addWidget(apply, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
            layout.addStretch(1)
            return page

        def _build_settings_page(self):
            page, layout = self._page("Настройки", "Настройки интерфейса и сохранение параметров RVC.")
            card = layout.itemAt(0).widget()
            card_layout = card.layout()
            self.remember_settings = QtWidgets.QCheckBox("Сохранять выбранные модель и аудиоустройства")
            self.remember_settings.setChecked(True)
            card_layout.addWidget(self.remember_settings)
            save = QtWidgets.QPushButton("Сохранить настройки сейчас")
            save.setObjectName("outlineButton")
            save.clicked.connect(lambda: self._save_settings(show_message=True))
            card_layout.addWidget(save, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
            layout.addStretch(1)
            return page

        def _build_logs_page(self):
            page, layout = self._page("Логи", "Здесь отображаются действия приложения и результаты запуска.")
            card = layout.itemAt(0).widget()
            card_layout = card.layout()
            self.log_view = QtWidgets.QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setMinimumHeight(280)
            self.log_view.setStyleSheet("background:#0C1220; border:1px solid #273249; border-radius:7px; color:#BFC9DC; padding:8px;")
            card_layout.addWidget(self.log_view)
            clear = QtWidgets.QPushButton("Очистить логи")
            clear.clicked.connect(self.log_view.clear)
            card_layout.addWidget(clear, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
            self._log("Deep Voice Live готов к работе.")
            layout.addStretch(1)
            return page

        def _build_about_page(self):
            page, layout = self._page("О программе", "Deep Voice Live by Syrex — локальный RVC-конвертер голоса в реальном времени.")
            card = layout.itemAt(0).widget()
            card.layout().addWidget(QtWidgets.QLabel("Версия интерфейса: 1.0.0\nДвижок: Retrieval-based Voice Conversion\nВычисления выполняются локально на GPU."))
            layout.addStretch(1)
            return page

        def _switch_page(self, index):
            self.pages.setCurrentIndex(index)
            for button_index, button in enumerate(self.nav_buttons):
                active = button_index == index
                button.setChecked(active)
                button.update()

        def _configure_shortcuts(self):
            for shortcut in getattr(self, "shortcuts", []):
                shortcut.deleteLater()
            self.shortcuts = []
            start_sequence = self.start_hotkey.keySequence() if hasattr(self, "start_hotkey") else QtGui.QKeySequence("Ctrl+Shift+S")
            stop_sequence = self.stop_hotkey.keySequence() if hasattr(self, "stop_hotkey") else QtGui.QKeySequence("Ctrl+Shift+X")
            for sequence, handler in ((start_sequence, self._start_live), (stop_sequence, self._stop_live)):
                shortcut = QtGui.QShortcut(sequence, self.ui)
                shortcut.activated.connect(handler)
                self.shortcuts.append(shortcut)
            self._log(f"Горячие клавиши применены: старт {start_sequence.toString()}, стоп {stop_sequence.toString()}.")

        def _profile_dir(self):
            path = os.path.join(now_dir, "profiles")
            os.makedirs(path, exist_ok=True)
            return path

        def _settings_dict(self):
            return {
                "pth_path": self.pth_field.text().strip(), "index_path": self.index_field.text().strip(),
                "sg_hostapi": self.host_combo.currentText(), "sg_wasapi_exclusive": self.wasapi_box.isChecked(),
                "sg_input_device": self.input_combo.currentText(), "sg_output_device": self.output_combo.currentText(),
                "sr_type": "sr_model" if self.rate_model.isChecked() else "sr_device",
                "threhold": self.controls["threhold"].value(), "pitch": self.controls["pitch"].value(),
                "formant": self.controls["formant"].value(), "index_rate": self.controls["index_rate"].value(),
                "rms_mix_rate": self.controls["rms_mix_rate"].value(), "block_time": self.controls["block_time"].value(),
                "crossfade_length": self.controls["crossfade_length"].value(), "extra_time": self.controls["extra_time"].value(),
                "I_noise_reduce": self.in_noise.isChecked(), "O_noise_reduce": self.out_noise.isChecked(),
                "f0method": "pm" if self.pm_radio.isChecked() else "fcpe" if self.fcpe_radio.isChecked() else "rmvpe",
            }

        def _save_settings(self, show_message=False):
            with open(realtime_config_path, "w", encoding="utf-8") as config_file:
                json.dump(self._settings_dict(), config_file, ensure_ascii=False, indent=2)
            self._log("Настройки сохранены.")
            if show_message:
                QtWidgets.QMessageBox.information(self.ui, "Deep Voice Live", "Настройки сохранены.")

        def _refresh_profiles(self):
            if not hasattr(self, "profile_list"):
                return
            self.profile_list.clear()
            for filename in sorted(os.listdir(self._profile_dir())):
                if filename.lower().endswith(".json"):
                    self.profile_list.addItem(os.path.splitext(filename)[0])

        def _save_profile(self):
            name = self.profile_name.text().strip()
            if not name:
                QtWidgets.QMessageBox.warning(self.ui, "Профиль", "Введите название профиля.")
                return
            safe_name = re.sub(r'[^\w. -]+', '_', name).strip('. ')
            if not safe_name:
                return
            with open(os.path.join(self._profile_dir(), safe_name + ".json"), "w", encoding="utf-8") as profile_file:
                json.dump(self._settings_dict(), profile_file, ensure_ascii=False, indent=2)
            self.profile_name.clear()
            self._refresh_profiles()
            self._log(f"Профиль «{safe_name}» сохранён.")

        def _load_profile(self, item):
            if item is None:
                return
            filename = os.path.join(self._profile_dir(), item.text() + ".json")
            try:
                with open(filename, encoding="utf-8") as profile_file:
                    settings = json.load(profile_file)
                self.pth_field.setText(settings.get("pth_path", ""))
                self.index_field.setText(settings.get("index_path", ""))
                for key, control in self.controls.items():
                    if key in settings:
                        control.setValue(float(settings[key]))
                self.in_noise.setChecked(settings.get("I_noise_reduce", False))
                self.out_noise.setChecked(settings.get("O_noise_reduce", False))
                {"pm": self.pm_radio, "rmvpe": self.rmvpe_radio, "fcpe": self.fcpe_radio}.get(settings.get("f0method", "rmvpe"), self.rmvpe_radio).setChecked(True)
                self._log(f"Профиль «{item.text()}» загружен.")
            except Exception as error:
                QtWidgets.QMessageBox.critical(self.ui, "Профиль", str(error))

        def _delete_profile(self):
            item = self.profile_list.currentItem()
            if item is None:
                return
            path = os.path.join(self._profile_dir(), item.text() + ".json")
            try:
                os.remove(path)
                self._refresh_profiles()
                self._log(f"Профиль «{item.text()}» удалён.")
            except OSError as error:
                QtWidgets.QMessageBox.critical(self.ui, "Профиль", str(error))

        def _log(self, message):
            if hasattr(self, "log_view"):
                self.log_view.appendPlainText(time.strftime("%H:%M:%S") + "  " + message)

        def _load_dropped_files(self, paths):
            loaded = []
            for path in paths:
                if path.lower().endswith(".pth"):
                    self.pth_field.setText(path)
                    loaded.append("модель .pth")
                elif path.lower().endswith(".index"):
                    self.index_field.setText(path)
                    loaded.append("индекс .index")
            if loaded:
                self._log("Загружено перетаскиванием: " + ", ".join(loaded) + ".")
            else:
                QtWidgets.QMessageBox.warning(self.ui, "Файлы модели", "Перетащите файл .pth и/или .index.")

        def _refresh_devices(self, hostapi_name=None):
            selected = hostapi_name or self.host_combo.currentText()
            self.update_devices(selected)
            for combo, values in ((self.host_combo, self.hostapis), (self.input_combo, self.input_devices), (self.output_combo, self.output_devices)):
                old = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(values)
                if old in values:
                    combo.setCurrentText(old)
                combo.blockSignals(False)

        def _apply_values(self):
            pth_path = self.pth_field.text().strip()
            index_path = self.index_field.text().strip()
            if not os.path.isfile(pth_path) or not os.path.isfile(index_path):
                raise ValueError("Выберите существующие файлы модели .pth и индекса .index.")
            self.set_devices(self.input_combo.currentText(), self.output_combo.currentText())
            self.gui_config.pth_path = pth_path
            self.gui_config.index_path = index_path
            self.gui_config.sg_hostapi = self.host_combo.currentText()
            self.gui_config.sg_wasapi_exclusive = self.wasapi_box.isChecked()
            self.gui_config.sg_input_device = self.input_combo.currentText()
            self.gui_config.sg_output_device = self.output_combo.currentText()
            self.gui_config.sr_type = "sr_model" if self.rate_model.isChecked() else "sr_device"
            for key, control in self.controls.items():
                self._hot_update(key, control.value())
            self.gui_config.I_noise_reduce = self.in_noise.isChecked()
            self.gui_config.O_noise_reduce = self.out_noise.isChecked()
            self.gui_config.f0method = "pm" if self.pm_radio.isChecked() else "fcpe" if self.fcpe_radio.isChecked() else "rmvpe"
            if not hasattr(self, "remember_settings") or self.remember_settings.isChecked():
                self._save_settings()

        def _test_microphone(self):
            if hasattr(self, "stream") and self.stream is not None:
                self._stop_live()
                return
            self.monitor_radio.setChecked(True)
            self._start_live()

        def _start_live(self):
            try:
                self._apply_values()
                self.start_vc()
                delay = self.stream.latency[-1] + self.gui_config.block_time + self.gui_config.crossfade_time + 0.01
                self.delay_label.setText(str(int(round(delay * 1000))))
                self.live_button.setText("●  LIVE: преобразование активно")
                self.live_button.setEnabled(False)
                self._log("Конвертация аудио запущена.")
            except Exception as error:
                self.stop_stream()
                self._log("Ошибка запуска: " + str(error))
                QtWidgets.QMessageBox.critical(self.ui, "Не удалось запустить", str(error))

        def _stop_live(self):
            self.stop_stream()
            self.live_button.setEnabled(True)
            self.live_button.setText("▶  Начать конвертацию аудио")
            self.delay_label.setText("—")
            self.infer_label.setText("—")
            self._log("Конвертация аудио остановлена.")

        def _set_function(self):
            sender = self.qt_app.sender()
            if sender is self.monitor_radio and self.monitor_radio.isChecked():
                self.convert_radio.blockSignals(True)
                self.convert_radio.setChecked(False)
                self.convert_radio.blockSignals(False)
            elif sender is self.convert_radio and self.convert_radio.isChecked():
                self.monitor_radio.blockSignals(True)
                self.monitor_radio.setChecked(False)
                self.monitor_radio.blockSignals(False)
            elif not self.monitor_radio.isChecked() and not self.convert_radio.isChecked():
                self.convert_radio.blockSignals(True)
                self.convert_radio.setChecked(True)
                self.convert_radio.blockSignals(False)
            self.function = "im" if self.monitor_radio.isChecked() else "vc"

        def _set_infer_time(self, value):
            self.signals.infer_time.emit(value)

        def _set_input_level(self, value):
            self.signals.input_level.emit(value)

        def _update_signal(self, value):
            self.waveform.set_level(value)
            if hasattr(self, "input_meter"):
                self.input_meter.set_level(value)
            db = 20 * np.log10(max(value, 1e-5))
            self.db_label.setText(f"{db:.0f} dB")
            if hasattr(self, "mic_level_label"):
                self.mic_level_label.setText(f"{db:.0f} dB")

    class ReferenceGUI(GUI):
        """Connects the supplied PySide6 reference UI to the existing RVC engine."""

        _row_keys = (
            "threhold", "pitch", "formant", "index_rate", "rms_mix_rate",
            "block_time", "crossfade_length", "extra_time",
        )

        def launcher(self):
            data = self.load()
            self._logs = ["Deep Voice Live запущен."]
            self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            self.qt_app.setApplicationName("Deep Voice Live")
            self.ui = ReferenceMainWindow()
            self.rows = self.ui.findChildren(ReferenceSliderRow)
            self.controls = {key: row.spin for key, row in zip(self._row_keys, self.rows)}
            for key, control in self.controls.items():
                if key in data:
                    control.setValue(float(data[key]))
            for key, row in zip(self._row_keys, self.rows):
                row.valueChanged.connect(lambda value, setting=key: self._hot_update(setting, value))

            self._set_combo(self.ui.host_api, self.hostapis, data.get("sg_hostapi", ""))
            self._refresh_reference_devices(data.get("sg_hostapi", ""), data)
            self.ui.pth_path.setText(data.get("pth_path", ""))
            self.ui.index_path.setText(data.get("index_path", ""))
            self.ui.wasapi_exclusive.setChecked(data.get("sg_wasapi_exclusive", False))
            self.ui.model_rate.setChecked(data.get("sr_type", "sr_model") == "sr_model")
            self.ui.device_rate.setChecked(not self.ui.model_rate.isChecked())
            self.ui.input_noise.setChecked(data.get("I_noise_reduce", False))
            self.ui.output_noise.setChecked(data.get("O_noise_reduce", False))
            for button in self.ui.pitch_group.buttons():
                button.setChecked(button.text().lower() == data.get("f0method", "rmvpe"))

            self.signals = UiSignals()
            self.signals.infer_time.connect(lambda value: self.ui.processing_spin.setValue(value))
            self.signals.input_level.connect(self._update_reference_level)
            self.signals.audio_samples.connect(self.ui.input_waveform.feed_samples)
            self.ui.host_api.currentTextChanged.connect(lambda name: self._refresh_reference_devices(name))
            self.ui.devices_refreshed.connect(lambda: self._refresh_reference_devices(self.ui.host_api.currentText()))
            self.ui.device_changed.connect(self._on_reference_device_changed)
            self.ui.model_path_changed.connect(lambda extension, path: self._save_reference_settings())
            self.ui.conversion_started.connect(self._start_reference_live)
            self.ui.conversion_stopped.connect(self._stop_reference_live)
            self.ui.microphone_test_requested.connect(self._start_reference_microphone_test)
            self.ui.settings_reset.connect(self._reset_reference_controls)
            self.ui.external_link_requested.connect(self._show_reference_link)
            self._connect_reference_navigation()
            self.qt_app.aboutToQuit.connect(self._save_reference_settings)
            self._last_studio_command_id = None
            if os.environ.get("DEEP_LIVE_STUDIO_EMBED") == "1":
                # Build a native Qt window but keep it off-screen until the
                # Deep Live Studio host re-parents it into the Voice tab.
                self.ui.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
                self.ui.winId()
                self.ui.show()
                self.ui.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False)
                self._studio_command_timer = QtCore.QTimer(self.ui)
                self._studio_command_timer.timeout.connect(self._read_studio_command)
                self._studio_command_timer.start(250)
            else:
                self.ui.show()
            self.qt_app.exec()

        def _read_studio_command(self):
            """Receive start/stop requests from the in-process Studio panel."""
            command_file = os.path.join(now_dir, "configs", "studio_command.json")
            try:
                with open(command_file, "r", encoding="utf-8") as handle:
                    command = json.load(handle)
                command_id = command.get("id")
                if not command_id or command_id == self._last_studio_command_id:
                    return
                self._last_studio_command_id = command_id
                action = command.get("action")
                if action == "start":
                    self._start_reference_live()
                elif action == "stop":
                    self._stop_reference_live()
                elif action == "mic_test":
                    self._start_reference_microphone_test()
            except (OSError, ValueError, TypeError):
                return

        @staticmethod
        def _set_combo(combo, values, selected):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            if selected in values:
                combo.setCurrentText(selected)
            combo.blockSignals(False)

        def _refresh_reference_devices(self, hostapi_name=None, saved=None):
            selected = hostapi_name or self.ui.host_api.currentText()
            self.update_devices(selected)
            saved = saved or {}
            current_input = saved.get("sg_input_device", self.ui.input_device.currentText())
            current_output = saved.get("sg_output_device", self.ui.output_device.currentText())
            self._set_combo(self.ui.input_device, self.input_devices, current_input)
            self._set_combo(self.ui.output_device, self.output_devices, current_output)

        def _hot_update(self, key, value):
            setattr(self.gui_config, key if key != "crossfade_length" else "crossfade_time", value)
            if not hasattr(self, "rvc"):
                return
            if key == "pitch":
                self.rvc.change_key(value)
            elif key == "formant":
                self.rvc.change_formant(value)
            elif key == "index_rate":
                self.rvc.change_index_rate(value)

        def _reference_settings(self):
            f0 = next((button.text().lower() for button in self.ui.pitch_group.buttons() if button.isChecked()), "rmvpe")
            return {
                "pth_path": self.ui.pth_path.text().strip(),
                "index_path": self.ui.index_path.text().strip(),
                "sg_hostapi": self.ui.host_api.currentText(),
                "sg_wasapi_exclusive": self.ui.wasapi_exclusive.isChecked(),
                "sg_input_device": self.ui.input_device.currentText(),
                "sg_output_device": self.ui.output_device.currentText(),
                "sr_type": "sr_model" if self.ui.model_rate.isChecked() else "sr_device",
                "threhold": self.controls["threhold"].value(),
                "pitch": self.controls["pitch"].value(),
                "formant": self.controls["formant"].value(),
                "index_rate": self.controls["index_rate"].value(),
                "rms_mix_rate": self.controls["rms_mix_rate"].value(),
                "block_time": self.controls["block_time"].value(),
                "crossfade_length": self.controls["crossfade_length"].value(),
                "extra_time": self.controls["extra_time"].value(),
                "I_noise_reduce": self.ui.input_noise.isChecked(),
                "O_noise_reduce": self.ui.output_noise.isChecked(),
                "f0method": f0,
            }

        def _save_reference_settings(self):
            try:
                with open(realtime_config_path, "w", encoding="utf-8") as config_file:
                    json.dump(self._reference_settings(), config_file, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _apply_reference_values(self):
            settings = self._reference_settings()
            if not os.path.isfile(settings["pth_path"]) or not os.path.isfile(settings["index_path"]):
                raise ValueError("Выберите существующие файлы модели .pth и индекса .index.")
            self.set_devices(settings["sg_input_device"], settings["sg_output_device"])
            for key, value in settings.items():
                if key == "crossfade_length":
                    self.gui_config.crossfade_time = value
                else:
                    setattr(self.gui_config, key, value)
            self.function = "im" if self.ui.monitor_toggle.isChecked() and not self.ui.convert_toggle.isChecked() else "vc"
            self._save_reference_settings()

        def _start_reference_live(self):
            try:
                # The microphone-test action deliberately enables input monitoring
                # only.  The main conversion button must always switch back to RVC
                # output, otherwise users hear the original microphone signal.
                self.ui.convert_toggle.setChecked(True)
                self._apply_reference_values()
                self.start_vc()
                self.ui.mic_test_button.setText("▶  Проверить звук")
                self.ui.mic_test_button.setProperty("testing", False)
                delay = self.stream.latency[-1] + self.gui_config.block_time + self.gui_config.crossfade_time + 0.01
                self.ui.latency_spin.setValue(int(round(delay * 1000)))
                self._append_log("Конвертация аудио запущена.")
            except Exception as error:
                details = traceback.format_exc()
                self._write_runtime_log("Start failed\n" + details)
                self.stop_stream()
                self.ui._conversion_active = False
                self.ui.input_waveform.set_running(False)
                self.ui.ready_label.setText("●  Ошибка запуска")
                self.ui.ready_label.setStyleSheet("color:#FF8395;font-size:11px;")
                self.ui.start_button.setEnabled(True)
                self.ui.stop_button.setEnabled(False)
                self._append_log("Ошибка запуска: " + str(error))
                QtWidgets.QMessageBox.critical(self.ui, "Не удалось запустить", str(error))

        def _stop_reference_live(self):
            self.stop_stream()
            self.ui.input_waveform.set_running(False)
            self._append_log("Конвертация аудио остановлена.")

        def _start_reference_microphone_test(self):
            if self.stream is not None:
                self._stop_reference_live()
                self.ui.mic_test_button.setText("▶  Проверить звук")
                self.ui.mic_test_button.setProperty("testing", False)
                self.ui.mic_test_button.style().unpolish(self.ui.mic_test_button)
                self.ui.mic_test_button.style().polish(self.ui.mic_test_button)
                return
            self.ui.monitor_toggle.setChecked(True)
            self.ui.convert_toggle.setChecked(False)
            try:
                self._apply_reference_values()
                self.start_vc()
            except Exception as error:
                self.stop_stream()
                self.ui.input_waveform.set_running(False)
                self.ui.mic_test_button.setText("▶  Проверить звук")
                self.ui.mic_test_button.setProperty("testing", False)
                self.ui.mic_test_button.style().unpolish(self.ui.mic_test_button)
                self.ui.mic_test_button.style().polish(self.ui.mic_test_button)
                QtWidgets.QMessageBox.critical(self.ui, "Не удалось запустить тест", str(error))

        def _on_reference_device_changed(self, input_device, output_device, hostapi_name):
            self._save_reference_settings()
            if self.stream is None:
                self.ui.ready_label.setText("●  Готов к запуску")
                self.ui.ready_label.setStyleSheet("color:#80eab5;font-size:11px;")
            self._append_log("Устройства обновлены: вход — %s; выход — %s." % (input_device, output_device))

        def _reset_reference_controls(self):
            defaults = {"threhold": -60, "pitch": 0, "formant": 0, "index_rate": .7, "rms_mix_rate": 1, "block_time": .13, "crossfade_length": .08, "extra_time": 2}
            for key, value in defaults.items():
                self.controls[key].setValue(value)

        def _show_reference_link(self, name):
            actions = {"О программе": self._show_about, "Логи": self._show_logs,
                       "Профили": self._show_profiles, "Справка": self._show_help}
            actions.get(name, self._show_help)()

        def _append_log(self, message):
            self._logs.append(time.strftime("%H:%M:%S") + "  " + message)

        def _connect_reference_navigation(self):
            buttons = {
                "home_button": self._show_home,
                "model_button": self._show_model_loader,
                "profiles_button": self._show_profiles,
                "hotkeys_button": self._show_hotkeys,
                "mixer_button": self._show_mixer,
                "effects_button": self._show_effects,
                "logs_button": self._show_logs,
                "settings_button": self._show_settings,
                "about_button": self._show_about,
            }
            for object_name, handler in buttons.items():
                button = self.ui.findChild(QtWidgets.QPushButton, object_name)
                if button is not None:
                    button.clicked.connect(handler)
            self._configure_reference_shortcuts("Ctrl+Shift+S", "Ctrl+Shift+X")

        def _show_home(self):
            self.ui.findChild(QtWidgets.QScrollArea).verticalScrollBar().setValue(0)

        def _show_model_loader(self):
            self._show_home()
            self.ui.pth_path.setFocus()
            self._append_log("Открыт раздел загрузки модели.")

        def _dialog(self, title, width=520):
            dialog = QtWidgets.QDialog(self.ui)
            dialog.setWindowTitle(title)
            dialog.setMinimumWidth(width)
            dialog.setStyleSheet("QDialog{background:#101827;color:#EAF0FF;} QLabel{color:#EAF0FF;} QLineEdit,QListWidget,QPlainTextEdit,QKeySequenceEdit,QDoubleSpinBox{background:#152138;border:1px solid #35496b;border-radius:6px;padding:6px;color:#EAF0FF;} QPushButton{background:#1A2941;border:1px solid #7657ff;border-radius:6px;padding:7px 10px;color:#EEE9FF;} QPushButton:hover{background:#283a5b;}")
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(16, 16, 16, 16)
            return dialog, layout

        def _profile_dir(self):
            path = os.path.join(now_dir, "profiles")
            os.makedirs(path, exist_ok=True)
            return path

        def _show_profiles(self):
            dialog, layout = self._dialog("Профили")
            layout.addWidget(QtWidgets.QLabel("Сохранение и загрузка наборов параметров RVC."))
            name = QtWidgets.QLineEdit()
            name.setPlaceholderText("Название нового профиля")
            profiles = QtWidgets.QListWidget()
            profiles.setMinimumHeight(190)

            def refresh():
                profiles.clear()
                for filename in sorted(os.listdir(self._profile_dir())):
                    if filename.lower().endswith(".json"):
                        profiles.addItem(os.path.splitext(filename)[0])

            def save():
                safe = re.sub(r"[^\w. -]+", "_", name.text().strip()).strip(". ")
                if not safe:
                    QtWidgets.QMessageBox.warning(dialog, "Профиль", "Введите название профиля.")
                    return
                with open(os.path.join(self._profile_dir(), safe + ".json"), "w", encoding="utf-8") as file:
                    json.dump(self._reference_settings(), file, ensure_ascii=False, indent=2)
                name.clear(); refresh(); self._append_log(f"Профиль «{safe}» сохранён.")

            def load():
                item = profiles.currentItem()
                if item is None:
                    return
                try:
                    with open(os.path.join(self._profile_dir(), item.text() + ".json"), encoding="utf-8") as file:
                        data = json.load(file)
                    self.ui.pth_path.setText(data.get("pth_path", ""))
                    self.ui.index_path.setText(data.get("index_path", ""))
                    self._set_combo(self.ui.input_device, self.input_devices, data.get("sg_input_device", ""))
                    self._set_combo(self.ui.output_device, self.output_devices, data.get("sg_output_device", ""))
                    for key, control in self.controls.items():
                        if key in data:
                            control.setValue(float(data[key]))
                    self.ui.input_noise.setChecked(data.get("I_noise_reduce", False))
                    self.ui.output_noise.setChecked(data.get("O_noise_reduce", False))
                    method = data.get("f0method", "rmvpe")
                    next((button for button in self.ui.pitch_group.buttons() if button.text().lower() == method), self.ui.pitch_group.buttons()[0]).setChecked(True)
                    self._append_log(f"Профиль «{item.text()}» загружен.")
                except Exception as error:
                    QtWidgets.QMessageBox.critical(dialog, "Профиль", str(error))

            def remove():
                item = profiles.currentItem()
                if item is None:
                    return
                os.remove(os.path.join(self._profile_dir(), item.text() + ".json"))
                self._append_log(f"Профиль «{item.text()}» удалён."); refresh()

            row = QtWidgets.QHBoxLayout(); save_button = QtWidgets.QPushButton("Сохранить"); save_button.clicked.connect(save)
            row.addWidget(name, 1); row.addWidget(save_button); layout.addLayout(row); layout.addWidget(profiles)
            actions = QtWidgets.QHBoxLayout(); load_button = QtWidgets.QPushButton("Загрузить"); delete_button = QtWidgets.QPushButton("Удалить")
            load_button.clicked.connect(load); delete_button.clicked.connect(remove)
            actions.addWidget(load_button); actions.addWidget(delete_button); actions.addStretch(); layout.addLayout(actions)
            refresh(); dialog.exec()

        def _configure_reference_shortcuts(self, start, stop):
            for shortcut in getattr(self, "reference_shortcuts", []):
                shortcut.deleteLater()
            self.reference_shortcuts = []
            for sequence, handler in ((start, self._start_shortcut), (stop, self._stop_reference_live)):
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self.ui)
                shortcut.activated.connect(handler)
                self.reference_shortcuts.append(shortcut)

        def _start_shortcut(self):
            if self.stream is None:
                self.ui.start_conversion()

        def _show_hotkeys(self):
            dialog, layout = self._dialog("Горячие клавиши")
            layout.addWidget(QtWidgets.QLabel("Работают, когда Deep Voice Live находится в фокусе."))
            start = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence("Ctrl+Shift+S"))
            stop = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence("Ctrl+Shift+X"))
            form = QtWidgets.QFormLayout(); form.addRow("Запуск конвертации", start); form.addRow("Остановка", stop); layout.addLayout(form)
            apply = QtWidgets.QPushButton("Применить")
            apply.clicked.connect(lambda: (self._configure_reference_shortcuts(start.keySequence().toString(), stop.keySequence().toString()), self._append_log("Горячие клавиши применены."), dialog.accept()))
            layout.addWidget(apply); dialog.exec()

        def _show_mixer(self):
            dialog, layout = self._dialog("Микшер")
            layout.addWidget(QtWidgets.QLabel("Баланс оригинального и преобразованного голоса."))
            mix = QtWidgets.QDoubleSpinBox(); mix.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons); mix.setRange(0, 1); mix.setSingleStep(.05); mix.setDecimals(2); mix.setValue(self.controls["rms_mix_rate"].value())
            layout.addWidget(mix)
            apply = QtWidgets.QPushButton("Применить баланс")
            apply.clicked.connect(lambda: (self.controls["rms_mix_rate"].setValue(mix.value()), self._hot_update("rms_mix_rate", mix.value()), self._append_log("Настройка микшера изменена."), dialog.accept()))
            layout.addWidget(apply); dialog.exec()

        def _show_effects(self):
            dialog, layout = self._dialog("Эффекты")
            layout.addWidget(QtWidgets.QLabel("Шумоподавление применяется движком RVC при запуске конвертации."))
            input_noise = QtWidgets.QCheckBox("Уменьшение входного шума"); input_noise.setChecked(self.ui.input_noise.isChecked())
            output_noise = QtWidgets.QCheckBox("Уменьшение выходного шума"); output_noise.setChecked(self.ui.output_noise.isChecked())
            layout.addWidget(input_noise); layout.addWidget(output_noise)
            apply = QtWidgets.QPushButton("Применить эффекты")
            apply.clicked.connect(lambda: (self.ui.input_noise.setChecked(input_noise.isChecked()), self.ui.output_noise.setChecked(output_noise.isChecked()), self._append_log("Эффекты обновлены."), dialog.accept()))
            layout.addWidget(apply); dialog.exec()

        def _show_logs(self):
            dialog, layout = self._dialog("Логи", 650)
            view = QtWidgets.QPlainTextEdit("\n".join(self._logs)); view.setReadOnly(True); view.setMinimumHeight(300)
            layout.addWidget(view); clear = QtWidgets.QPushButton("Очистить")
            clear.clicked.connect(lambda: (self._logs.clear(), view.clear())); layout.addWidget(clear); dialog.exec()

        def _show_settings(self):
            dialog, layout = self._dialog("Настройки")
            layout.addWidget(QtWidgets.QLabel("Настройки текущей модели и устройств сохраняются локально в configs/config.json."))
            refresh = QtWidgets.QPushButton("Обновить список устройств"); refresh.clicked.connect(lambda: self._refresh_reference_devices(self.ui.host_api.currentText()))
            save = QtWidgets.QPushButton("Сохранить настройки"); save.clicked.connect(lambda: (self._save_reference_settings(), self._append_log("Настройки сохранены."), dialog.accept()))
            layout.addWidget(refresh); layout.addWidget(save); dialog.exec()

        def _show_about(self):
            QtWidgets.QMessageBox.information(self.ui, "О программе", "Deep Voice Live by Syrex\n\nЛокальный RVC-конвертер голоса в реальном времени.\nВерсия интерфейса: 1.0.0")

        def _show_help(self):
            QtWidgets.QMessageBox.information(self.ui, "Справка", "1. Выберите .pth и .index модели.\n2. Выберите вход и выход аудио.\n3. Нажмите «Начать конвертацию аудио».\n\nВсе действия выполняются локально; внешние сайты не открываются.")

        def _set_infer_time(self, value):
            self.signals.infer_time.emit(value)

        def _set_input_level(self, value):
            self.signals.input_level.emit(value)

        def _set_audio_samples(self, samples):
            self.signals.audio_samples.emit(samples)

        def _update_reference_level(self, value):
            db = 20 * np.log10(max(value, 1e-5))
            self.ui.signal_db.setText(f"{db:.0f} dB")
            self.ui.input_waveform.set_audio_level(value)

    gui = ReferenceGUI()
