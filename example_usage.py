"""
Example usage of the Document Classification OpenEnv Environment
Demonstrates various ways to interact with the environment
"""

import numpy as np
from environment import DocumentClassificationEnv
from grading import BaselineAgent, AgentGrader


def example_basic_usage():
    """Example 1: Basic environment usage"""
    print("=" * 70)
    print("Example 1: Basic Environment Usage")
    print("=" * 70)
    
    # Create environment
    env = DocumentClassificationEnv(task_difficulty="easy")
    print(f"✓ Created environment: {env.task_difficulty}")
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"✓ Reset environment")
    print(f"  Initial document: {obs['document_id']}")
    print(f"  Content preview: {obs['content'][:60]}...")
    
    # Take a few steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {env.CATEGORY_MAPS['easy'][action]}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Episode Accuracy: {info['episode_accuracy']:.3f}")
        
        if done:
            print(f"  Episode Complete!")
            break
    
    print(f"\nTotal Reward: {total_reward:.3f}")


def example_state_inspection():
    """Example 2: Inspecting environment state"""
    print("\n" + "=" * 70)
    print("Example 2: Environment State Inspection")
    print("=" * 70)
    
    env = DocumentClassificationEnv(task_difficulty="medium")
    obs, _ = env.reset()
    
    # Take a few steps
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
    
    # Get and inspect state
    state = env.state()
    print("Environment State:")
    print(f"  Current document index: {state['current_document_index']}")
    print(f"  Total documents: {state['total_documents']}")
    print(f"  Episode reward: {state['episode_reward_total']:.3f}")
    print(f"  Accuracy so far: {state['episode_accuracy']:.3f}")
    print(f"  Avg processing time: {state['average_processing_time_ms']:.2f}ms")
    
    print(f"\nCurrent Observation:")
    print(f"  Document ID: {state['current_observation']['document_id']}")
    print(f"  Word count: {state['current_observation']['word_count'][0]}")
    print(f"  Has urgency: {bool(state['current_observation']['has_urgency_markers'][0])}")


def example_baseline_agent():
    """Example 3: Using the baseline agent"""
    print("\n" + "=" * 70)
    print("Example 3: Baseline Agent Performance")
    print("=" * 70)
    
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n{difficulty.upper()} Task:")
        
        env = DocumentClassificationEnv(task_difficulty=difficulty)
        agent = BaselineAgent(difficulty)
        
        obs, _ = env.reset()
        
        total_reward = 0
        correct = 0
        total = 0
        
        for step in range(20):  # Run 20 steps
            action = agent.decide(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            if info['is_correct']:
                correct += 1
            total += 1
            
            if done:
                break
        
        accuracy = correct / total if total > 0 else 0
        print(f"  Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"  Total Reward: {total_reward:.3f}")
        print(f"  Avg Reward: {total_reward / total:.3f}")


def example_agent_grading():
    """Example 4: Grading an agent"""
    print("\n" + "=" * 70)
    print("Example 4: Agent Grading System")
    print("=" * 70)
    
    difficulty = "easy"
    
    # Create grader
    grader = AgentGrader(difficulty)
    
    # Create a simple custom agent
    def custom_agent(obs):
        """A simple custom agent that prefers 'Support' category"""
        # 70% of the time, choose Support (index 2 in easy)
        if np.random.random() < 0.7:
            return 2  # Support
        else:
            return np.random.randint(0, 5)  # Random action
    
    # Grade the custom agent
    score, metrics = grader.grade_agent(custom_agent, verbose=True)
    
    print(f"Final Score: {score:.4f}")


def example_difficulty_comparison():
    """Example 5: Comparing difficulty levels"""
    print("\n" + "=" * 70)
    print("Example 5: Difficulty Level Comparison")
    print("=" * 70)
    
    difficulties = ["easy", "medium", "hard"]
    
    print(f"\n{'Difficulty':<10} {'Categories':<12} {'Documents':<12} {'Time Limit':<15} {'Features':<10}")
    print("-" * 60)
    
    for diff in difficulties:
        env = DocumentClassificationEnv(task_difficulty=diff)
        config = env.task_config
        
        time_limit = config.get("time_limit", "None")
        time_str = f"{time_limit}s" if time_limit else "None"
        
        print(f"{diff:<10} {config['num_categories']:<12} {config['num_documents']:<12} {time_str:<15} {config['feature_dim']:<10}")


def example_seed_reproducibility():
    """Example 6: Demonstrating reproducibility with seeds"""
    print("\n" + "=" * 70)
    print("Example 6: Seed-based Reproducibility")
    print("=" * 70)
    
    # Create two environments with the same seed
    env1 = DocumentClassificationEnv(task_difficulty="easy", seed=42)
    env2 = DocumentClassificationEnv(task_difficulty="easy", seed=42)
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    print("Comparing two environments with seed=42:")
    print(f"  Env1 doc ID: {obs1['document_id']}")
    print(f"  Env2 doc ID: {obs2['document_id']}")
    print(f"  Match: {obs1['document_id'] == obs2['document_id']}")
    
    print(f"\n  Env1 content: {obs1['content'][:40]}...")
    print(f"  Env2 content: {obs2['content'][:40]}...")
    print(f"  Match: {obs1['content'] == obs2['content']}")
    
    print(f"\n  Feature vectors equal: {np.allclose(obs1['features'], obs2['features'])}")


def example_episode_summary():
    """Example 7: Getting episode summary"""
    print("\n" + "=" * 70)
    print("Example 7: Episode Summary")
    print("=" * 70)
    
    env = DocumentClassificationEnv(task_difficulty="easy")
    obs, _ = env.reset()
    
    # Run a full episode
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
    
    # Get summary from info
    summary = info.get("episode_summary", {})
    
    print("Episode Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    """Run all examples"""
    try:
        example_basic_usage()
        example_state_inspection()
        example_baseline_agent()
        example_agent_grading()
        example_difficulty_comparison()
        example_seed_reproducibility()
        example_episode_summary()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
