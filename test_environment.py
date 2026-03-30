"""
Test script to validate the Document Classification Environment
"""

import sys
from environment import DocumentClassificationEnv
from grading import BaselineAgent, AgentGrader


def test_environment_creation():
    """Test that environments can be created for all difficulties"""
    print("Testing environment creation...")
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = DocumentClassificationEnv(task_difficulty=difficulty)
            obs, info = env.reset()
            print(f"✓ {difficulty} environment created successfully")
            assert "features" in obs, f"Missing features in {difficulty} observation"
            assert obs["features"].shape[0] > 0, f"Invalid feature shape in {difficulty}"
        except Exception as e:
            print(f"✗ Failed to create {difficulty} environment: {e}")
            return False
    return True


def test_step_function():
    """Test that step function works correctly"""
    print("\nTesting step function...")
    env = DocumentClassificationEnv(task_difficulty="easy")
    obs, _ = env.reset()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert isinstance(done, bool), f"Done should be bool, got {type(done)}"
        print(f"✓ Step {i+1}: reward={reward:.3f}, accuracy={info['episode_accuracy']:.3f}")
        if done:
            break
    
    print(f"✓ Step function working correctly")
    return True


def test_state_function():
    """Test that state function works correctly"""
    print("\nTesting state function...")
    env = DocumentClassificationEnv(task_difficulty="easy")
    obs, _ = env.reset()
    
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
    
    state = env.state()
    assert "current_observation" in state, "Missing current_observation in state"
    assert "current_document_index" in state, "Missing current_document_index in state"
    assert "episode_reward_total" in state, "Missing episode_reward_total in state"
    print(f"✓ State function working correctly")
    print(f"  State keys: {list(state.keys())}")
    return True


def test_baseline_agent():
    """Test baseline agent on each difficulty"""
    print("\nTesting baseline agent...")
    for difficulty in ["easy"]:  # Test only easy for speed
        try:
            env = DocumentClassificationEnv(task_difficulty=difficulty)
            agent = BaselineAgent(difficulty)
            obs, _ = env.reset()
            
            for _ in range(5):
                action = agent.decide(obs)
                assert 0 <= action < env.num_categories, f"Invalid action: {action}"
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break
            
            print(f"✓ Baseline agent working on {difficulty}")
        except Exception as e:
            print(f"✗ Baseline agent failed on {difficulty}: {e}")
            return False
    return True


def test_grading():
    """Test the grading system"""
    print("\nTesting grading system...")
    try:
        grader = AgentGrader("easy")
        agent = BaselineAgent("easy")
        
        score, metrics = grader.grade_agent(agent.decide, verbose=False)
        
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert "accuracy" in metrics, "Missing accuracy in metrics"
        assert metrics["accuracy"] >= 0.0, "Negative accuracy"
        
        print(f"✓ Grading system working")
        print(f"  Score: {score:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Grading system failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Document Classification Environment - Test Suite")
    print("="*60)
    
    tests = [
        ("Environment Creation", test_environment_creation),
        ("Step Function", test_step_function),
        ("State Function", test_state_function),
        ("Baseline Agent", test_baseline_agent),
        ("Grading System", test_grading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
