# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf Runners.

Runners orchestrate the training loop for different RL algorithms:

- EmbodiedRunner: Standard PPO for embodied/robot tasks
- EmbodiedHILRunner: PPO with optional Human-in-the-Loop support
- HILSACRunner: SAC/RLPD with Human-in-the-Loop (off-policy)
- ReasoningRunner: PPO for reasoning/language model tasks
- AgentRunner: For agent/tool-use tasks
"""

# Import runners (lazy imports to avoid circular dependencies)
__all__ = [
    "EmbodiedRunner",
    "EmbodiedHILRunner", 
    "HILSACRunner",
    "HILSACConfig",
    "EmbodiedHILConfig",
]
