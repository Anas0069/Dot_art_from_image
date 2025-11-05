from DOT_ART_GENE import create_dot_art

# Your specific file paths
input_path = "D:\\Projects\\dot_art\\input_image.jpg"
output_path = "D:\\Projects\\dot_art\\dot_art_output.png"

try:
    print("🎨 Starting dot art generation...")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    # Generate dot art
    result = create_dot_art(
        image_path=input_path,
        output_path=output_path,
        style='classic',    # Black background, white dots
        quality='high'      # Maximum quality
    )
    
    print("\n✅ SUCCESS! Dot art created successfully!")
    print(f"📂 Check your output at: {output_path}")
    
except FileNotFoundError:
    print("❌ ERROR: Could not find input image!")
    print("Make sure the file exists at:", input_path)
except Exception as e:
    print(f"❌ ERROR: {str(e)}")