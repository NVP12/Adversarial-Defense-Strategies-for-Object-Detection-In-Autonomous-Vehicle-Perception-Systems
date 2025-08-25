import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .get_map import evaluate_map


class eval_map(Node):
    def __init__(self, node_name="mAP_calculation"):
        super().__init__(node_name)
        self.subscription = self.create_subscription(String, '/classification_completion_flag', self.trigger, 1)
        self.subscription = self.create_subscription(String, '/model_weights', self.weight_callback, 1)
        self.pub_completion = "/mAP_completion_flag" 
        self.publish_classification = self.create_publisher(String, self.pub_completion, 1)
        self.publish_classification.publish(String(data="not_complete"))
        
        
    def weight_callback(self,msg):
        self.weights = msg.data
        
    def trigger(self, msg):
        flag = String()
        flag.data = msg.data
        #this_dir = os.path.dirname(os.path.realpath(__file__))
        #print(this_dir)
        #txt_path = os.path.join(this_dir, weights_path)
        if (flag.data == "complete"):
            evaluate_map(test_txt_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/clean_text.txt', classes_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/model_data/coco_classes.txt', image_ext='.jpg', map_mode=0, model_weights=self.weights)
            self.publish_classification.publish(String(data="complete"))
            

            
            


def main(args=None):
    rclpy.init(args=args)
    node = eval_map("mAP_calculation")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()
