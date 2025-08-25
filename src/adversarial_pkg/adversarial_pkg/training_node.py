import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .get_map import evaluate_map
from .train import train_yolov4 

class trainingNode(Node):
    def __init__(self, node_name="training_loop"):
        super().__init__(node_name)
        self.subscription = self.create_subscription(String, '/SAM_completion_flag', self.trigger, 1)
        self.subscription = self.create_subscription(String, '/model_weights', self.weight_callback, 1)
        self.pub_completion = "/training_completion_flag" 
        self.publish_classification = self.create_publisher(String, self.pub_completion, 1)
        self.publish_classification.publish(String(data="not_complete"))
        
        
    def weight_callback(self, msg):
        self.weights = msg.data
        
        
    def trigger(self, msg):
        flag = String()
        flag.data = msg.data
        #this_dir = os.path.dirname(os.path.realpath(__file__))
        #print(this_dir)
        #txt_path = os.path.join(this_dir, weights_path)
        if (flag.data == "complete"):
            train_yolov4(Cuda=True,seed=11,distributed=False,sync_bn=False,fp16=False,classes_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/model_data/coco_classes.txt',anchors_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/model_data/yolo_anchors.txt',anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],model_path=self.weights,input_shape=[416, 416],pretrained=False,mosaic=False,mosaic_prob=0.5,mixup=False,mixup_prob=0.5,special_aug_ratio=0.7,label_smoothing=0,Init_Epoch=0,Freeze_Epoch=0,Freeze_batch_size=8,UnFreeze_Epoch=20,Unfreeze_batch_size=4,Freeze_Train=False, Init_lr=1e-4,optimizer_type="sgd",momentum=0.937,weight_decay=5e-4,lr_decay_type="cos",focal_loss=False,focal_alpha=0.25,focal_gamma=2,iou_type='ciou',save_period=10,save_dir='logs',eval_flag=True,eval_period=1,num_workers=4,train_annotation_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_train.txt',val_annotation_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_val.txt')
            
            self.publish_classification.publish(String(data="complete"))
            
            


def main(args=None):
    rclpy.init(args=args)
    node = trainingNode("training_loop")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()
