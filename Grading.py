from ultralytics import YOLO


# path file test cua BTC cho
source_test = 'E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Processed_data/testthu/'
# path file model AI m train xong luu ve may
model = YOLO("E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Models/best2.pt")
results = model.predict(source_test, save = False, stream = True)
import os

# path file predict.txt m luu vao de submit len web
output_file = "E:\\OneDrive - Hanoi University of Science and Technology\\HUST\\Competition\\AMIC\\Results\\results.txt"

with open('E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Results/a/result-1.txt', 'r') as f:
    bound = f.read()
    box = bound.split("\n")
    boxes_1 = [list(map(float, i.split(' '))) for i in box]

print(boxes_1)
def inside(box, x, y):
    if x >= box[1] and x <= box[3] and y >= box[2] and y <= box[4]:
        return True
    return False

with open(output_file, "w") as f:
    for result in results:
        image_name = os.path.basename(result.path)
        f.write(f"{image_name} \n")
        # result.show()
        sbd = 1
        mdt = 1
        num = 1
        data = []
        for box in result.boxes:
            class_id = int(box.cls)
            x = box.xywhn[0][0]
            y = box.xywhn[0][1]
            width = box.xywhn[0][2]
            height = box.xywhn[0][3]
            # print(box.xywhn)
            if class_id != 0:
                continue
            data.append([x ,y ,width ,height])
        new_data = sorted(data, key= lambda x :(x[1], x[0]))
        # for i in data:
        #     if inside(boxes_1[0], i[0], i[1]):
        #         f.write(f"SBD1 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[1], i[0], i[1]):
        #         f.write(f"SBD2 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[2], i[0], i[1]):
        #         f.write(f"SBD3 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[3], i[0], i[1]):
        #         f.write(f"SBD4 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[4], i[0], i[1]):
        #         f.write(f"SBD5 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[5], i[0], i[1]):
        #         f.write(f"SBD6 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[6], i[0], i[1]):
        #         f.write(f"MDT1 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[7], i[0], i[1]):
        #         f.write(f"MDT2 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for i in data:
        #     if inside(boxes_1[8], i[0], i[1]):
        #         f.write(f"MDT3 {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for j in range(40):
        #     for i in data:
        #         if inside(boxes_1[9+j], i[0], i[1]):
        #             f.write(f"1.{j+1} {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        d = {'1':'a', '2':'b', '3':'c', '0':'d'}
        for j in range(32):
            for i in data:
                if inside(boxes_1[49+j], i[0], i[1]):
                    r = d[str((j+1)%4)]
                    f.write(f"2.{j//4+1}.{r} {i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # for j in range(6):
        #     f.write(f"3.{j+1} \n")
        #     for i in data:
        #         if inside(boxes_1[81+j], i[0], i[1]):
        #             f.write(f"{i[0]:.6f},{i[1]:.6f},{i[2]:.6f},{i[3]:.6f} \n")
        # f.write("\n")

# python "E:/OneDrive - Hanoi University of Science and Technology/HUST/AI Hackathon/Model/Hyper-YOLOv1.1-main/Hyper-YOLOv1.1-main/detect.py" --source 'E:/OneDrive - Hanoi University of Science and Technology/HUST/AI Hackathon/public_test/pub test latest/cam_08_00500_jpg.rf.5ab59b5bcda1d1fad9131385c5d64fdb.jpg' --img 640 --device 0 --weights 'E:/OneDrive - Hanoi University of Science and Technology/HUST/AI Hackathon/Model/bestlatest.pt' --name yolov9_c_hyper_c_640_detect