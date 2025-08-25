import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .get_map import evaluate_map
from .teacher_fn import run_yolo_inference_from_list 
import time

class teacherNode(Node):
    def __init__(self, node_name="training_labels"):
        super().__init__(node_name)
        self.subscription = self.create_subscription(String, '/classification_completion_flag', self.trigger, 1)
        self.pub_completion = "/SAM_completion_flag" 
        self.publish_classification = self.create_publisher(String, self.pub_completion, 1)
        self.publish_classification.publish(String(data="not_complete"))
        
        
    def trigger(self, msg):
        flag = String()
        flag.data = msg.data
        #this_dir = os.path.dirname(os.path.realpath(__file__))
        #print(this_dir)
        #txt_path = os.path.join(this_dir, weights_path)
        if (flag.data == "complete"):
            run_yolo_inference_from_list(model_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/best.pt', image_list_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_sample.txt', output_txt_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_train.txt', output_txt_path2='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_val.txt')
            
            self.publish_classification.publish(String(data="complete"))
            self.reset()
    def reset(self):
        time.sleep(10)
        self.publish_classification.publish(String(data="not_complete"))
            
            


def main(args=None):
    rclpy.init(args=args)
    node = teacherNode("teacher_labels")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()
