import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import pytesseract
import requests
import json
import threading
from datetime import datetime
import socket
import time
from transformers import BlipProcessor, BlipForConditionalGeneration

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class SceneDescriber:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        try:
            print("🔄 Loading scene description model...")
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            print("✅ Scene description model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load scene model: {e}")
            self.processor = None
            self.model = None

    def describe_scene(self, image_path: str) -> str:
        """
        Generates a natural language description of an image.
        """
        if self.processor is None or self.model is None:
            return "Scene description model not available"
            
        try:
            # Open and convert the image to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Process the image for the model
            inputs = self.processor(image, return_tensors="pt")
            
            # Generate the description
            outputs = self.model.generate(**inputs)
            description = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            return f"Unable to describe scene: {str(e)}"


class BookDiscussionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Discussion Desktop App - Dark Theme")
        self.root.geometry("1200x700")
        
        # Dark theme colors
        self.bg_dark = "#1e1e1e"
        self.bg_darker = "#121212"
        self.bg_card = "#2d2d2d"
        self.text_primary = "#ffffff"
        self.text_secondary = "#b0b0b0"
        self.accent_color = "#bb86fc"
        self.accent_dark = "#3700b3"
        self.success_color = "#03dac6"
        self.error_color = "#cf6679"
        
        # Configure root background
        self.root.configure(bg=self.bg_dark)
        
        # Variables
        self.api_key = tk.StringVar()
        self.captured_image = None
        self.extracted_text = ""
        self.messages = []
        self.camera = None
        self.is_camera_active = False
        
        # Create UI
        self.create_api_key_screen()
        
    def create_api_key_screen(self):
        """Initial screen to enter API key"""
        self.api_frame = tk.Frame(self.root, bg=self.bg_dark)
        self.api_frame.pack(expand=True, fill='both', padx=50, pady=50)
        
        # Title
        title = tk.Label(
            self.api_frame, 
            text="📚 Book Discussion App", 
            font=('Arial', 28, 'bold'),
            bg=self.bg_dark,
            fg=self.text_primary
        )
        title.pack(pady=20)
        
        subtitle = tk.Label(
            self.api_frame,
            text="Capture, read, and discuss your books with AI",
            font=('Arial', 12),
            bg=self.bg_dark,
            fg=self.text_secondary
        )
        subtitle.pack(pady=10)
        
        # API Key input
        key_label = tk.Label(
            self.api_frame,
            text="Groq API Key:",
            font=('Arial', 11, 'bold'),
            bg=self.bg_dark,
            fg=self.text_primary
        )
        key_label.pack(pady=(30, 5))
        
        key_entry = tk.Entry(
            self.api_frame,
            textvariable=self.api_key,
            font=('Arial', 11),
            width=40,
            show='*',
            bg=self.bg_card,
            fg=self.text_primary,
            insertbackground=self.text_primary,  # Cursor color
            relief='flat',
            highlightthickness=1,
            highlightbackground=self.text_secondary,
            highlightcolor=self.accent_color
        )
        key_entry.pack(pady=5)
        
        # Start button
        start_btn = tk.Button(
            self.api_frame,
            text="Start Reading",
            font=('Arial', 12, 'bold'),
            bg=self.accent_color,
            fg=self.bg_dark,
            activebackground=self.accent_dark,
            activeforeground=self.text_primary,
            relief='flat',
            padx=40,
            pady=10,
            cursor='hand2',
            command=self.start_app
        )
        start_btn.pack(pady=20)
        
        # Instructions
        info_frame = tk.Frame(self.api_frame, bg=self.bg_card, relief='solid', borderwidth=1)
        info_frame.pack(pady=20, padx=20, fill='x')
        
        info_text = tk.Label(
            info_frame,
            text="Get your free API key:\n\n"
                 "1. Visit console.groq.com\n"
                 "2. Sign up for a free account\n"
                 "3. Generate an API key\n"
                 "4. Paste it above",
            font=('Arial', 10),
            bg=self.bg_card,
            fg=self.text_primary,
            justify='left'
        )
        info_text.pack(padx=20, pady=15)
        
    def start_app(self):
        """Transition to main app"""
        if not self.api_key.get():
            messagebox.showerror("Error", "Please enter your Groq API key")
            return
        
        self.api_frame.destroy()
        self.create_main_screen()
        
    def create_main_screen(self):
        """Main application screen"""
        # Top bar
        top_bar = tk.Frame(self.root, bg=self.bg_card, height=60)
        top_bar.pack(fill='x')
        top_bar.pack_propagate(False)
        
        title = tk.Label(
            top_bar,
            text="📚 Book Discussion",
            font=('Arial', 16, 'bold'),
            bg=self.bg_card,
            fg=self.text_primary
        )
        title.pack(side='left', padx=20, pady=15)
        
        new_page_btn = tk.Button(
            top_bar,
            text="New Page",
            font=('Arial', 10, 'bold'),
            bg=self.accent_color,
            fg=self.bg_dark,
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2',
            command=self.reset_app
        )
        new_page_btn.pack(side='right', padx=20, pady=15)
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.bg_dark)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Left side - Camera
        left_frame = tk.Frame(main_frame, bg=self.bg_card, relief='solid', borderwidth=1)
        left_frame.pack(side='left', expand=True, fill='both', padx=5, pady=5)
        
        camera_label = tk.Label(
            left_frame,
            text="Capture Book Page",
            font=('Arial', 14, 'bold'),
            bg=self.bg_card,
            fg=self.text_primary
        )
        camera_label.pack(pady=10)
        
        self.camera_canvas = tk.Canvas(left_frame, width=500, height=400, bg=self.bg_darker)
        self.camera_canvas.pack(pady=10, padx=10)
        
        # Camera buttons
        btn_frame = tk.Frame(left_frame, bg=self.bg_card)
        btn_frame.pack(pady=10)
        
        self.start_cam_btn = tk.Button(
            btn_frame,
            text="📷 Start Camera",
            font=('Arial', 11, 'bold'),
            bg=self.accent_color,
            fg=self.bg_dark,
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self.start_camera
        )
        self.start_cam_btn.pack(side='left', padx=5)
        
        self.capture_btn = tk.Button(
            btn_frame,
            text="📸 Capture",
            font=('Arial', 11, 'bold'),
            bg=self.success_color,
            fg=self.bg_dark,
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self.capture_image,
            state='disabled'
        )
        self.capture_btn.pack(side='left', padx=5)
        
        self.status_label = tk.Label(
            left_frame,
            text="Click 'Start Camera' to begin",
            font=('Arial', 10),
            bg=self.bg_card,
            fg=self.text_secondary
        )
        self.status_label.pack(pady=5)
        
        # Right side - Discussion
        right_frame = tk.Frame(main_frame, bg=self.bg_card, relief='solid', borderwidth=1)
        right_frame.pack(side='right', expand=True, fill='both', padx=5, pady=5)
        
        discussion_label = tk.Label(
            right_frame,
            text="💬 Discussion",
            font=('Arial', 14, 'bold'),
            bg=self.bg_card,
            fg=self.text_primary
        )
        discussion_label.pack(pady=10)
        
        # Messages area
        self.messages_area = scrolledtext.ScrolledText(
            right_frame,
            font=('Arial', 10),
            wrap=tk.WORD,
            bg=self.bg_darker,
            fg=self.text_primary,
            relief='flat',
            padx=10,
            pady=10,
            insertbackground=self.text_primary  # Cursor color
        )
        self.messages_area.pack(expand=True, fill='both', padx=10, pady=5)
        self.messages_area.config(state='disabled')
        
        # Configure text tags for styling
        self.messages_area.tag_config('user', foreground=self.success_color, font=('Arial', 10, 'bold'))
        self.messages_area.tag_config('assistant', foreground=self.accent_color, font=('Arial', 10, 'bold'))
        self.messages_area.tag_config('content', foreground=self.text_primary, font=('Arial', 10))
        
        # Input area
        input_frame = tk.Frame(right_frame, bg=self.bg_card)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        self.user_input = tk.Entry(
            input_frame,
            font=('Arial', 11),
            relief='flat',
            borderwidth=1,
            bg=self.bg_darker,
            fg=self.text_primary,
            insertbackground=self.text_primary,
            highlightthickness=1,
            highlightbackground=self.text_secondary,
            highlightcolor=self.accent_color
        )
        self.user_input.pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.user_input.bind('<Return>', lambda e: self.send_message())
        
        send_btn = tk.Button(
            input_frame,
            text="Send",
            font=('Arial', 10, 'bold'),
            bg=self.accent_color,
            fg=self.bg_dark,
            relief='flat',
            padx=20,
            pady=5,
            cursor='hand2',
            command=self.send_message
        )
        send_btn.pack(side='right')
        
    def start_camera(self):
        """Start camera feed"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
            
            self.is_camera_active = True
            self.start_cam_btn.config(state='disabled')
            self.capture_btn.config(state='normal')
            self.status_label.config(text="Camera active - Position your book page")
            self.update_camera_feed()
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            
    def update_camera_feed(self):
        """Update camera feed in canvas"""
        if self.is_camera_active and self.camera:
            ret, frame = self.camera.read()
            if ret:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to fit canvas
                frame_resized = cv2.resize(frame_rgb, (500, 400))
                # Convert to PIL Image
                img = Image.fromarray(frame_resized)
                # Convert to ImageTk
                imgtk = ImageTk.PhotoImage(image=img)
                # Update canvas
                self.camera_canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.camera_canvas.image = imgtk
                
            # Schedule next update
            self.root.after(10, self.update_camera_feed)
            
    def capture_image(self):
        """Capture current frame"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.captured_image = frame
                self.stop_camera()
                
                # Display captured image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (500, 400))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.camera_canvas.image = imgtk
                
                self.status_label.config(text="Processing image...")
                
                # Perform OCR in separate thread
                threading.Thread(target=self.perform_ocr, daemon=True).start()
                
    def stop_camera(self):
        """Stop camera feed"""
        self.is_camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_cam_btn.config(state='normal')
        self.capture_btn.config(state='disabled')
        
    def perform_ocr(self):
        """Extract text from captured image"""
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
            # Perform OCR
            text = pytesseract.image_to_string(img_rgb)
            self.extracted_text = text
            
            # Update UI in main thread
            self.root.after(0, self.on_ocr_complete, text)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("OCR Error", str(e)))
            self.root.after(0, lambda: self.status_label.config(text="OCR failed"))
            
    def on_ocr_complete(self, text):
        """Handle OCR completion"""
        if text.strip():
            self.status_label.config(text="Text extracted successfully!", fg=self.success_color)
            
            # Start discussion
            preview = text[:300] + ("..." if len(text) > 300 else "")
            initial_msg = (
                f"I've read the text from your book page. Here's what I found:\n\n"
                f'"{preview}"\n\n'
                f"What would you like to discuss about this passage?"
            )
            self.display_message("assistant", initial_msg)
            
        else:
            self.status_label.config(text="No text detected. Try again with better lighting.", fg=self.error_color)
            result = messagebox.askyesno(
                        "No Text",
                        "No text detected in the image. I can try to describe the scene. Do you want to try?"
                    )
            if result:
                # Generate scene description in a separate thread to avoid blocking
                threading.Thread(target=self.generate_scene_description, daemon=True).start()
            else:
                # User declined scene description
                self.display_message("assistant", "I couldn't find any text in this image. Please try capturing a book page with clearer text and better lighting.")

    def generate_scene_description(self):
        """Generate scene description in background thread"""
        try:
            temp_image_path = "temp_capture.jpg"
            cv2.imwrite(temp_image_path, self.captured_image)
            
            # Initialize the describer and generate a scene description
            describer = SceneDescriber()
            scene_description = describer.describe_scene(temp_image_path)
            
            # Update the extracted text with the scene description
            self.extracted_text = f"I couldn't find any text in this image, but I can describe the scene. It appears to be: {scene_description}"
            
            # Update UI in main thread
            self.root.after(0, self.on_scene_description_complete, scene_description)
            
        except Exception as e:
            error_msg = f"Scene description failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.root.after(0, lambda: self.display_message("assistant", "I couldn't generate a scene description. Please try capturing a different image."))

    def on_scene_description_complete(self, scene_description):
        """Handle scene description completion"""
        message = f"I couldn't find any text in this image, but I can describe what I see: {scene_description}."
        self.display_message("assistant", message)

        # Let the user know they can discuss it
        self.display_message(
            "assistant",
            "Would you like to discuss what this scene might mean or represent?"
        )
        self.status_label.config(text="Scene description complete!", fg=self.success_color)
            
    def send_message(self):
        """Send user message and get AI response"""
        user_msg = self.user_input.get().strip()
        if not user_msg or not self.extracted_text:
            return
        
        self.user_input.delete(0, tk.END)
        self.display_message("user", user_msg)
        
        # Get AI response in separate thread
        threading.Thread(
            target=self.get_ai_response,
            args=(user_msg,),
            daemon=True
        ).start()
        
    def get_ai_response(self, user_msg):
        """Get response from Groq API"""
        try:
            # Build conversation history
            conversation = [
                {
                    "role": "system",
                    "content": "You are a thoughtful book discussion companion. Help users explore and understand the text they have captured. Ask insightful questions, provide analysis, and engage in meaningful dialogue about the content."
                },
                {
                    "role": "user", 
                    "content": f"Here is the text from the book page:\n\n{self.extracted_text}\n\nNow let's discuss it."
                }
            ]
            
            # Add message history
            for msg in self.messages:
                conversation.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current message
            conversation.append({"role": "user", "content": user_msg})
            
            # Call Groq API
            print("🤖 Contacting Groq API...")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key.get()}"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": conversation,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )
            
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data["choices"][0]["message"]["content"]
                print(f"🤖 AI response received: {ai_response[:100]}...")
                self.root.after(0, self.display_message, "assistant", ai_response)
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                # Show error in chat instead of messagebox to avoid Tkinter issues
                self.root.after(0, self.display_message, "assistant", f"I encountered an error: {error_msg}")
                
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"❌ {error_msg}")
            # Show error in chat instead of messagebox
            self.root.after(0, self.display_message, "assistant", f"I couldn't connect to the AI service. Please check your API key and internet connection.")
            
    def display_message(self, role, content):
        """Display message in chat area"""
        self.messages.append({"role": role, "content": content})
        
        self.messages_area.config(state='normal')
        
        # Add spacing
        if len(self.messages) > 1:
            self.messages_area.insert(tk.END, "\n\n")
        
        # Add role label
        role_label = "You: " if role == "user" else "AI: "
        self.messages_area.insert(tk.END, role_label, role)
        self.messages_area.insert(tk.END, "\n")
        
        # Add message content
        self.messages_area.insert(tk.END, content, 'content')
        
        self.messages_area.config(state='disabled')
        self.messages_area.see(tk.END)
        
    def reset_app(self):
        """Reset to capture new page"""
        self.stop_camera()
        self.captured_image = None
        self.extracted_text = ""
        self.messages = []
        
        # Clear messages area
        self.messages_area.config(state='normal')
        self.messages_area.delete(1.0, tk.END)
        self.messages_area.config(state='disabled')
        
        # Clear canvas
        self.camera_canvas.delete("all")
        
        # Reset status
        self.status_label.config(text="Click 'Start Camera' to begin", fg=self.text_secondary)
        self.user_input.delete(0, tk.END)

def main():
    root = tk.Tk()
    app = BookDiscussionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()