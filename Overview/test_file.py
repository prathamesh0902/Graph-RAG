from code_file import Calculator

# Create instance
calc = Calculator()

# Test only add() method manually
try:
    result = calc.add(3, 5)
    assert result == 8, f"Expected 8, got {result}"
    print("✅ test_add_positive_numbers passed")

    result = calc.add(-2, -4)
    assert result == -6, f"Expected -6, got {result}"
    print("✅ test_add_negative_numbers passed")

    # Test invalid type
    try:
        calc.add("a", 5)
    except TypeError:
        print("✅ test_add_invalid_type passed")
    else:
        print("❌ test_add_invalid_type failed")

except AssertionError as e:
    print(f"❌ Test failed: {e}")