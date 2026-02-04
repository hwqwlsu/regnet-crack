import sys
import argparse
import cv2
import os
import time
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
                             QGroupBox, QStatusBar, QFrame, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PyQt5.QtCore import Qt, QTimer

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default=r"runs\weights\5_regnet_imse_impiou\weights\best.pt", type=str)
parser.add_argument('--conf_thre', type=float, default=0.3)
parser.add_argument('--iou_thre', type=float, default=0.5)
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'

STYLE = """
QMainWindow { background: #1a1a1a; }
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3a3a3a,stop:1 #2a2a2a);
    color: #00d4aa; border: 1px solid #00d4aa; border-radius: 4px;
    padding: 10px 15px; font: bold 13px; min-width: 100px;
}
QPushButton:hover { background: #00d4aa; color: #1a1a1a; }
QPushButton:pressed { background: #00a080; }
QTableWidget {
    background: #252525; border: 1px solid #00d4aa; color: #e0e0e0;
    gridline-color: #3a3a3a; selection-background-color: #00d4aa;
}
QHeaderView::section { background: #2a2a2a; color: #00d4aa; padding: 8px; border: none; font: bold; }
QTabWidget::pane { border: 1px solid #00d4aa; background: #1a1a1a; }
QTabBar::tab { background: #2a2a2a; color: #888; padding: 10px 25px; border: 1px solid #3a3a3a; }
QTabBar::tab:selected { background: #1a1a1a; color: #00d4aa; border-bottom: 2px solid #00d4aa; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 4px; margin-top: 10px; padding-top: 15px; background: #252525; color: #00d4aa; font: bold 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLabel#info { background: #252525; border: 1px solid #3a3a3a; border-radius: 4px; padding: 8px; color: #e0e0e0; font: 12px; }
QLabel#value { color: #00d4aa; font: bold 14px; background: transparent; border: none; }
QLabel#title { color: #00d4aa; font: bold 11px; background: transparent; border: none; }
"""


def get_color(idx):
    colors = [QColor(0, 212, 170), QColor(255, 193, 7), QColor(244, 67, 54), QColor(33, 150, 243)]
    return colors[idx % len(colors)]


class Detector:
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        from ultralytics.models.yolo.detect import DetectionPredictor
        self.predictor = DetectionPredictor(overrides={'model': weight_path, 'conf': conf_threshold, 'iou': iou_threshold, 'device': device, 'save': False, 'verbose': False})
        self.predictor.setup_model(weight_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.names = self.predictor.model.names

    def detect_image(self, img_bgr):
        self.predictor.args.conf = self.conf_threshold
        self.predictor.args.iou = self.iou_threshold
        results = list(self.predictor.stream_inference([img_bgr]))
        detection_data = []
        for idx in range(len(results[0].boxes.cls)):
            cls_id = int(results[0].boxes.cls[idx])
            conf = float(results[0].boxes.conf[idx])
            box = results[0].boxes.xyxy[idx].cpu().numpy().astype('uint32')
            label = self.names[cls_id]
            detection_data.append((label, conf, box))
            xmin, ymin, xmax, ymax = box
            color = get_color(cls_id)
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (color.blue(), color.green(), color.red()), 2)
            cv2.rectangle(img_bgr, (xmin, ymin - 22), (xmin + len(f'{label} {conf:.2f}') * 9, ymin), (color.blue(), color.green(), color.red()), -1)
            cv2.putText(img_bgr, f'{label} {conf:.2f}', (xmin + 3, ymin - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return img_bgr, detection_data


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("裂缝智能检测系统")
        self.setMinimumSize(1400, 850)
        self.setStyleSheet(STYLE)
        self.detector = None
        self.class_counts = {}
        self.total_count = 0
        self.video_timer = QTimer()
        self.camera_timer = QTimer()
        self.cap = None
        self.start_time = time.time()
        self.frame_count = 0
        self.setup_ui()
        QTimer.singleShot(100, self.load_model)

    def load_model(self):
        self.status_bar.showMessage("正在加载模型...")
        QApplication.processEvents()
        self.detector = Detector(opt.weights, opt.conf_thre, opt.iou_thre)
        self.status_bar.showMessage("就绪")

    def setup_ui(self):
        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setFixedWidth(240)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 标题
        title = QLabel("裂缝智能检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#00d4aa; font:bold 16px; padding:15px; background:#252525; border:1px solid #00d4aa; border-radius:4px;")
        left_layout.addWidget(title)

        # 按钮
        self.btn_image = QPushButton("图像检测")
        self.btn_video = QPushButton("视频检测")
        self.btn_camera = QPushButton("实时监控")
        self.btn_export = QPushButton("导出报告")
        for btn in [self.btn_image, self.btn_video, self.btn_camera, self.btn_export]:
            btn.setCursor(Qt.PointingHandCursor)
            left_layout.addWidget(btn)

        # 系统状态
        status_group = QGroupBox("系统状态")
        status_layout = QGridLayout(status_group)
        status_layout.setSpacing(8)

        self.lbl_device = self._create_info_pair("运算设备", "GPU" if torch.cuda.is_available() else "CPU")
        self.lbl_model = self._create_info_pair("检测模型", opt.weights.split('/')[-1][:12])
        self.lbl_conf = self._create_info_pair("置信度", f"{opt.conf_thre:.2f}")
        self.lbl_iou = self._create_info_pair("IoU阈值", f"{opt.iou_thre:.2f}")

        for i, (t, v) in enumerate([self.lbl_device, self.lbl_model, self.lbl_conf, self.lbl_iou]):
            status_layout.addWidget(t, i, 0)
            status_layout.addWidget(v, i, 1)
        left_layout.addWidget(status_group)

        # 检测统计
        stats_group = QGroupBox("检测统计")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setSpacing(8)

        self.lbl_total = self._create_info_pair("检测总数", "0")
        self.lbl_fps = self._create_info_pair("帧率", "0.0")
        self.lbl_time = self._create_info_pair("运行时间", "00:00:00")

        for i, (t, v) in enumerate([self.lbl_total, self.lbl_fps, self.lbl_time]):
            stats_layout.addWidget(t, i, 0)
            stats_layout.addWidget(v, i, 1)
        left_layout.addWidget(stats_group)
        left_layout.addStretch()

        # 右侧主区域
        tab_widget = QTabWidget()

        # 检测画面
        self.image_label = QLabel("请选择图像或视频")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#252525; border:1px solid #3a3a3a; color:#666; font:14px;")
        self.image_label.setMinimumSize(850, 620)
        tab_widget.addTab(self.image_label, "检测画面")

        # 统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["裂缝类型", "检测数量"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        tab_widget.addTab(self.stats_table, "统计分析")

        # 历史记录
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["时间", "模式", "类型", "置信度", "位置"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        tab_widget.addTab(self.history_table, "检测记录")

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(tab_widget, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background:#252525; color:#888; border-top:1px solid #3a3a3a;")
        self.status_bar.showMessage("就绪")
        self.setStatusBar(self.status_bar)

        # 连接信号
        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.toggle_video)
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_export.clicked.connect(self.export_records)
        self.video_timer.timeout.connect(self.process_video)
        self.camera_timer.timeout.connect(self.process_camera)

        # 时间更新
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)

    def _create_info_pair(self, title, value):
        t = QLabel(title)
        t.setObjectName("title")
        v = QLabel(value)
        v.setObjectName("value")
        v.setAlignment(Qt.AlignRight)
        return t, v

    def update_time(self):
        elapsed = int(time.time() - self.start_time)
        self.lbl_time[1].setText(f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")

    def open_image(self):
        if not self.detector:
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                img, detections = self.detector.detect_image(img)
                self.update_ui(img, detections, "IMAGE")

    def toggle_video(self):
        if not self.detector:
            return
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.btn_video.setText("视频检测")
            if self.cap: self.cap.release()
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
            if file_name:
                self.cap = cv2.VideoCapture(file_name)
                if self.cap.isOpened():
                    self.video_timer.start(33)
                    self.btn_video.setText("停止")

    def toggle_camera(self):
        if not self.detector:
            return
        if self.camera_timer.isActive():
            self.camera_timer.stop()
            self.btn_camera.setText("实时监控")
            if self.cap: self.cap.release()
        else:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_timer.start(33)
                self.btn_camera.setText("停止")

    def process_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame, detections = self.detector.detect_image(frame)
            self.update_ui(frame, detections, "VIDEO")

    def process_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame, detections = self.detector.detect_image(frame)
            self.update_ui(frame, detections, "CAMERA")

    def update_ui(self, frame, detections, mode):
        self.frame_count += 1
        self.total_count += len(detections)

        # 显示图像
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新统计
        self.lbl_total[1].setText(str(self.total_count))
        elapsed = time.time() - self.start_time
        self.lbl_fps[1].setText(f"{self.frame_count / max(elapsed, 1):.1f}")

        self.update_stats(detections)
        self.update_history(detections, mode)
        self.status_bar.showMessage(f"检测到: {len(detections)} 处裂缝 | 分辨率: {w}x{h} | 模式: {mode}")

    def update_stats(self, detections):
        self.class_counts.clear()
        for det in detections:
            self.class_counts[det[0]] = self.class_counts.get(det[0], 0) + 1
        self.stats_table.setRowCount(len(self.class_counts))
        for row, (label, count) in enumerate(self.class_counts.items()):
            for col, text in enumerate([label, str(count)]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(row, col, item)

    def update_history(self, detections, mode):
        timestamp = time.strftime("%H:%M:%S")
        for label, conf, box in detections:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            for col, text in enumerate([timestamp, mode, label, f"{conf:.3f}", f"({box[0]},{box[1]})-({box[2]},{box[3]})"]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, col, item)
            if self.history_table.rowCount() > 500:
                self.history_table.removeRow(0)

    def export_records(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "导出报告", f"裂缝检测报告_{time.strftime('%Y%m%d_%H%M%S')}.csv", "CSV文件 (*.csv)")
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("时间,模式,类型,置信度,位置\n")
                for row in range(self.history_table.rowCount()):
                    f.write(",".join([self.history_table.item(row, col).text() for col in range(5)]) + "\n")

    def closeEvent(self, event):
        if self.cap: self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
