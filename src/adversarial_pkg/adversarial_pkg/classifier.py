import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PIL import Image
from torchvision import transforms
import torch
from .magnet_inference import magnet_detect
import os

class Classifier(Node):
    def __init__(self, node_name = "classifier_node"):
        super().__init__(node_name)
        #self.subscription = self.create_subscription(String, '/input_trigger', self.trigger, 1)
        self.pub_completion = "/classification_completion_flag" 
        self.publish_classification = self.create_publisher(String, self.pub_completion, 1)
        self.publish_classification.publish(String(data="not_complete"))
        
    
    
# Image preprocessing
to_tensor = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])


def process_file(node,input_txt, clean_txt, adv_txt):
    adv_train_txt = '/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_sample.txt'

    # Step 1: Clear files before starting
    open(clean_txt, 'w').close()
    open(adv_txt, 'w').close()
    open(adv_train_txt, 'w').close()

    # Step 2: Begin processing
    with open(input_txt, 'r') as f:
        for line in f:
            original_line = line.strip()
            if not original_line:
                continue

            parts = original_line.split()
            image_path = parts[0]
            bbox_entries = parts[1:]

            # Extract image name
            image_name = os.path.basename(image_path).replace('.jpg', '')

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = to_tensor(image)

            # Load weights for magnet detector
            weights_path = 'best_detector.pt'
            this_dir = os.path.dirname(os.path.realpath(__file__))
            ckpt_path = os.path.join(this_dir, weights_path)

            is_adv = magnet_detect(image_tensor, ckpt_path)

            # Append result to clean/adv list
            with open(adv_txt if is_adv else clean_txt, 'a') as out_file:
                out_file.write(image_name + '\n')

            # Append full line to adv_train.txt if adversarial
            if is_adv:
                with open(adv_train_txt, 'a') as atf:
                    atf.write(original_line + '\n')
                    #atf.write(image_path + '\n')
                    
                    
    node.publish_classification.publish(String(data="complete"))


    
    


def main(args = None):
    rclpy.init(args=args)
    node = Classifier("classifier_node")
    process_file(node=node,input_txt='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/train_kitti.txt', clean_txt='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/clean_text.txt', adv_txt='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/adv_text.txt') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
    
    

if __name__ == '__main__':
    main()
