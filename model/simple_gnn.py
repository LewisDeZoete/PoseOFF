import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonFlowActionClassifier(nn.Module):
    def __init__(self, num_features=168, num_frames=300, num_keypoints=17, num_people=2, num_classes=10):
        super(SkeletonFlowActionClassifier, self).__init__()
        
        # Temporal convolution across frames for each keypoint and person, adjusted for higher input features
        self.temporal_conv1 = nn.Conv2d(num_features, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
        self.temporal_conv2 = nn.Conv2d(128, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
        
        # Optional: Feature reduction to manage high dimensionality if necessary
        self.feature_reduce = nn.Conv2d(256, 128, kernel_size=1) if num_features > 64 else None

        # Spatial convolution across keypoints
        self.spatial_conv1 = nn.Conv2d(128, 256, kernel_size=(1, num_keypoints))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * num_frames * num_people, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x is of shape (num_features, num_frames, num_keypoints, num_people)
        
        # Reshape x to (batch_size, num_features, num_frames, num_keypoints * num_people)
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4))
        
        # Temporal convolution
        x = F.relu(self.temporal_conv1(x))  # Shape: (batch_size, 128, num_frames, num_keypoints * num_people)
        x = F.relu(self.temporal_conv2(x))  # Shape: (batch_size, 256, num_frames, num_keypoints * num_people)
        
        # Feature reduction if needed
        if self.feature_reduce:
            x = F.relu(self.feature_reduce(x))  # Shape adjusted accordingly if feature_reduce is used

        # Spatial convolution
        x = self.spatial_conv1(x)            # Shape: (batch_size, 256, num_frames, num_people)
        
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)            # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Example usage
model = SkeletonFlowActionClassifier(num_features=168, num_frames=300, num_keypoints=17, num_people=2, num_classes=10)

# Dummy input for testing
x = torch.randn(1, 168, 300, 17, 2)  # Example input with (num_features, num_frames, num_keypoints, num_people)

# Forward pass
output = model(x)
print("Output shape:", output.shape)