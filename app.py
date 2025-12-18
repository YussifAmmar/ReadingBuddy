import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import requests
import threading
import os
import time
from transformers import BlipProcessor, BlipForConditionalGeneration


class SceneDescriber:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        try:
            print("🔄 Loading scene description model...")
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            print("✅ Scene description loaded!")
        except Exception as e:
            print(f"❌ Failed to load scene model: {e}")
            self.processor = self.model = None

    def describe_scene(self, image_path):
        if not self.processor or not self.model:
            return "Scene description model not available"
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Unable to describe scene: {e}"


class BookDiscussionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Discussion Desktop App - Dark Theme")
        self.root.geometry("1200x700")
        
        # Dark theme colors
        self.colors = {
            'bg_dark': "#1e1e1e", 'bg_darker': "#121212", 'bg_card': "#2d2d2d",
            'text_primary': "#ffffff", 'text_secondary': "#b0b0b0",
            'accent': "#bb86fc", 'accent_dark': "#3700b3",
            'success': "#03dac6", 'error': "#cf6679"
        }
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Variables
        self.api_key = tk.StringVar()
        self.hf_api_key = tk.StringVar()
        self.captured_image = None
        self.extracted_text = ""
        self.messages = []
        self.camera = None
        self.is_camera_active = False
        
        self.create_api_key_screen()
    
    def _create_label(self, parent, text, font_size=11, bold=False, fg_key='text_primary', bg_key='bg_dark'):
        return tk.Label(parent, text=text, font=('Arial', font_size, 'bold' if bold else 'normal'),
                       bg=self.colors[bg_key], fg=self.colors[fg_key])
    
    def _create_entry(self, parent, textvariable, show='*', width=40, bg_key='bg_card'):
        return tk.Entry(parent, textvariable=textvariable, font=('Arial', 11), width=width, show=show,
                       bg=self.colors[bg_key], fg=self.colors['text_primary'],
                       insertbackground=self.colors['text_primary'], relief='flat', highlightthickness=1,
                       highlightbackground=self.colors['text_secondary'], highlightcolor=self.colors['accent'])
    
    def _create_button(self, parent, text, command, bg_key='accent', padx=20, pady=8, state='normal'):
        return tk.Button(parent, text=text, font=('Arial', 11, 'bold'), bg=self.colors[bg_key],
                        fg=self.colors['bg_dark'], relief='flat', padx=padx, pady=pady,
                        cursor='hand2', command=command, state=state,
                        activebackground=self.colors['accent_dark'], activeforeground=self.colors['text_primary'])
        
    def create_api_key_screen(self):
        self.api_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        self.api_frame.pack(expand=True, fill='both', padx=50, pady=50)
        
        # Title and subtitle
        self._create_label(self.api_frame, "📚 Book Discussion App", 28, True).pack(pady=20)
        self._create_label(self.api_frame, "Capture, read, and discuss your books with AI", 12, fg_key='text_secondary').pack(pady=10)
        
        # Groq API Key
        self._create_label(self.api_frame, "Groq API Key:", 11, True).pack(pady=(30, 5))
        self._create_entry(self.api_frame, self.api_key).pack(pady=5)
        
        # Hugging Face API Key
        self._create_label(self.api_frame, "Hugging Face API Key (for OCR):", 11, True).pack(pady=(20, 5))
        self._create_entry(self.api_frame, self.hf_api_key).pack(pady=5)
        
        # Info box
        info_frame = tk.Frame(self.api_frame, bg=self.colors['bg_card'])
        info_frame.pack(pady=20, padx=20, fill='x')
        tk.Label(info_frame, text="API Keys Required:\n\n"
                "1. Groq API Key: console.groq.com → Generate API key\n"
                "2. Hugging Face: huggingface.co/settings/tokens → Create token",
                font=('Arial', 10), bg=self.colors['bg_card'], fg=self.colors['text_primary'],
                justify='left').pack(padx=20, pady=15)
        
        # Start button
        self._create_button(self.api_frame, "Start Reading", self.start_app, padx=40, pady=10).pack(pady=20)
        
    def start_app(self):
        if not self.api_key.get():
            return messagebox.showerror("Error", "Please enter your Groq API key")
        if not self.hf_api_key.get():
            return messagebox.showerror("Error", "Please enter your Hugging Face API key")
        
        os.environ['HF_API_KEY'] = self.hf_api_key.get()
        self.api_frame.destroy()
        self.create_main_screen()
        
    def create_main_screen(self):
        # Top bar
        top_bar = tk.Frame(self.root, bg=self.colors['bg_card'], height=60)
        top_bar.pack(fill='x')
        top_bar.pack_propagate(False)
        
        self._create_label(top_bar, "📚 Book Discussion", 16, True, bg_key='bg_card').pack(side='left', padx=20, pady=15)
        self._create_button(top_bar, "New Page", self.reset_app, padx=15, pady=5).pack(side='right', padx=20, pady=15)
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Left side - Camera
        left_frame = tk.Frame(main_frame, bg=self.colors['bg_card'], relief='solid', borderwidth=1)
        left_frame.pack(side='left', expand=True, fill='both', padx=5, pady=5)
        
        self._create_label(left_frame, "Capture Book Page", 14, True, bg_key='bg_card').pack(pady=10)
        self.camera_canvas = tk.Canvas(left_frame, width=500, height=400, bg=self.colors['bg_darker'])
        self.camera_canvas.pack(pady=10, padx=10)
        
        btn_frame = tk.Frame(left_frame, bg=self.colors['bg_card'])
        btn_frame.pack(pady=10)
        self.start_cam_btn = self._create_button(btn_frame, "📷 Start Camera", self.start_camera)
        self.start_cam_btn.pack(side='left', padx=5)
        self.capture_btn = self._create_button(btn_frame, "📸 Capture", self.capture_image, bg_key='success', state='disabled')
        self.capture_btn.pack(side='left', padx=5)
        
        self.status_label = self._create_label(left_frame, "Click 'Start Camera' to begin", 10, fg_key='text_secondary', bg_key='bg_card')
        self.status_label.pack(pady=5)
        
        # Right side - Discussion
        right_frame = tk.Frame(main_frame, bg=self.colors['bg_card'], relief='solid', borderwidth=1)
        right_frame.pack(side='right', expand=True, fill='both', padx=5, pady=5)
        
        self._create_label(right_frame, "💬 Discussion", 14, True, bg_key='bg_card').pack(pady=10)
        
        self.messages_area = scrolledtext.ScrolledText(right_frame, font=('Arial', 10), wrap=tk.WORD,
            bg=self.colors['bg_darker'], fg=self.colors['text_primary'], relief='flat', padx=10, pady=10,
            insertbackground=self.colors['text_primary'])
        self.messages_area.pack(expand=True, fill='both', padx=10, pady=5)
        self.messages_area.config(state='disabled')
        self.messages_area.tag_config('user', foreground=self.colors['success'], font=('Arial', 10, 'bold'))
        self.messages_area.tag_config('assistant', foreground=self.colors['accent'], font=('Arial', 10, 'bold'))
        self.messages_area.tag_config('content', foreground=self.colors['text_primary'], font=('Arial', 10))
        
        input_frame = tk.Frame(right_frame, bg=self.colors['bg_card'])
        input_frame.pack(fill='x', padx=10, pady=10)
        
        self.user_input = tk.Entry(input_frame, font=('Arial', 11), relief='flat', borderwidth=1,
            bg=self.colors['bg_darker'], fg=self.colors['text_primary'], insertbackground=self.colors['text_primary'],
            highlightthickness=1, highlightbackground=self.colors['text_secondary'], highlightcolor=self.colors['accent'])
        self.user_input.pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.user_input.bind('<Return>', lambda e: self.send_message())
        self._create_button(input_frame, "Send", self.send_message, padx=20, pady=5).pack(side='right')
        
    def start_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return messagebox.showerror("Error", "Could not access camera")
            self.is_camera_active = True
            self.start_cam_btn.config(state='disabled')
            self.capture_btn.config(state='normal')
            self.status_label.config(text="Camera active - Position your book page")
            self.update_camera_feed()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            
    def update_camera_feed(self):
        if self.is_camera_active and self.camera:
            ret, frame = self.camera.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (500, 400))
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                self.camera_canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.camera_canvas.image = imgtk
            self.root.after(10, self.update_camera_feed)
            
    def capture_image(self):
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.captured_image = frame
                self.stop_camera()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(frame_rgb, (500, 400))))
                self.camera_canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.camera_canvas.image = imgtk
                self.status_label.config(text="Processing image...")
                threading.Thread(target=self.perform_ocr, daemon=True).start()
                
    def stop_camera(self):
        self.is_camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_cam_btn.config(state='normal')
        self.capture_btn.config(state='disabled')
        
    def perform_ocr(self):
        """Extract text using Hugging Face Vision-Language model"""
        import base64
        temp_path = "temp_ocr.jpg"
        try:
            cv2.imwrite(temp_path, self.captured_image)
            self.root.after(0, lambda: self.status_label.config(text="Extracting text..."))
            
            hf_api_key = os.getenv('HF_API_KEY', '')
            
            if not hf_api_key:
                print("❌ HF_API_KEY not found")
                self.root.after(0, lambda: self.status_label.config(text="API key missing", fg=self.colors['error']))
                self.root.after(0, self.handle_ocr_failure)
                return
            
            # Convert image to base64
            with open(temp_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            print("🔄 Calling Hugging Face Vision API...")
            
            # Use the Inference API with chat completion format for VL models
            api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-3B-Instruct"
            headers = {
                "Authorization": f"Bearer {hf_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": {
                    "image": image_base64,
                    "text": "Extract and return all the text you can read from this image. Only output the extracted text, nothing else."
                }
            }
            
            # Try up to 3 times (for model loading)
            for attempt in range(3):
                try:
                    print(f"  Attempt {attempt + 1}/3...")
                    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
                    
                    print(f"  Response status: {resp.status_code}")
                    print(f"  Response: {resp.text[:500]}")
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        # Parse response based on format
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get("generated_text", "") or result[0].get("text", "")
                        elif isinstance(result, dict):
                            text = result.get("generated_text", "") or result.get("text", "") or result.get("output", "")
                        else:
                            text = str(result)
                        
                        if text.strip():
                            print(f"✅ Success: {len(text)} chars extracted")
                            self.extracted_text = text.strip()
                            self.root.after(0, self.on_ocr_complete, text.strip())
                            return
                        else:
                            print("  No text in response")
                            
                    elif resp.status_code == 503:
                        print("  Model loading, waiting 10s...")
                        time.sleep(10)
                        continue
                    else:
                        print(f"  Error: {resp.text[:200]}")
                        
                except requests.exceptions.Timeout:
                    print(f"  Timeout on attempt {attempt + 1}")
                except Exception as e:
                    print(f"  Error: {e}")
            
            # If we get here, OCR failed
            print("❌ All OCR attempts failed")
            self.root.after(0, lambda: self.status_label.config(text="OCR failed", fg=self.colors['error']))
            self.root.after(0, self.handle_ocr_failure)
            
        except Exception as e:
            print(f"❌ OCR failed: {e}")
            self.root.after(0, lambda: self.status_label.config(text="OCR failed", fg=self.colors['error']))
            self.root.after(0, self.handle_ocr_failure)
        finally:
            try: os.path.exists(temp_path) and os.remove(temp_path)
            except: pass


            
    def handle_ocr_failure(self):
        """Handle complete OCR failure"""
        if messagebox.askyesno("OCR Failed", "Could not extract text.\n\nTry scene description instead?"):
            threading.Thread(target=self.generate_scene_description, daemon=True).start()
        else:
            self.display_message("assistant", "I couldn't extract text. Try:\n1. Better lighting\n2. Steady camera\n3. Clear, focused text")
            
    def on_ocr_complete(self, text):
        if text.strip():
            self.status_label.config(text="Text extracted!", fg=self.colors['success'])
            preview = text[:300] + ("..." if len(text) > 300 else "")
            self.display_message("assistant", f'I\'ve read the text:\n\n"{preview}"\n\nWhat would you like to discuss?')
        else:
            self.status_label.config(text="No text detected", fg=self.colors['error'])
            if messagebox.askyesno("No Text", "No text detected. Try scene description?"):
                threading.Thread(target=self.generate_scene_description, daemon=True).start()
            else:
                self.display_message("assistant", "I couldn't find text. Please try with better lighting.")

    def generate_scene_description(self):
        try:
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, self.captured_image)
            describer = SceneDescriber()
            description = describer.describe_scene(temp_path)
            self.extracted_text = f"Scene: {description}"
            self.root.after(0, self.on_scene_complete, description)
        except Exception as e:
            print(f"❌ Scene description failed: {e}")
            self.root.after(0, lambda: self.display_message("assistant", "Couldn't generate scene description."))

    def on_scene_complete(self, description):
        self.display_message("assistant", f"I see: {description}.\n\nWould you like to discuss this?")
        self.status_label.config(text="Scene description complete!", fg=self.colors['success'])
            
    def send_message(self):
        user_msg = self.user_input.get().strip()
        if not user_msg or not self.extracted_text: return
        self.user_input.delete(0, tk.END)
        self.display_message("user", user_msg)
        threading.Thread(target=self.get_ai_response, args=(user_msg,), daemon=True).start()
        
    def get_ai_response(self, user_msg):
        try:
            conversation = [
                {"role": "system", "content": "You are a thoughtful book discussion companion. Help users explore and understand captured text."},
                {"role": "user", "content": f"Text from book:\n\n{self.extracted_text}\n\nLet's discuss."}
            ] + [{"role": m["role"], "content": m["content"]} for m in self.messages] + [{"role": "user", "content": user_msg}]
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key.get()}"},
                json={"model": "llama-3.1-8b-instant", "messages": conversation, "temperature": 0.7, "max_tokens": 1000})
            
            if response.status_code == 200:
                ai_response = response.json()["choices"][0]["message"]["content"]
                self.root.after(0, self.display_message, "assistant", ai_response)
            else:
                self.root.after(0, self.display_message, "assistant", f"API Error: {response.status_code}")
        except Exception as e:
            self.root.after(0, self.display_message, "assistant", f"Connection error. Check API key and internet.")
            
    def display_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.messages_area.config(state='normal')
        if len(self.messages) > 1: self.messages_area.insert(tk.END, "\n\n")
        self.messages_area.insert(tk.END, f"{'You' if role == 'user' else 'AI'}: ", role)
        self.messages_area.insert(tk.END, f"\n{content}", 'content')
        self.messages_area.config(state='disabled')
        self.messages_area.see(tk.END)
        
    def reset_app(self):
        self.stop_camera()
        self.captured_image = self.extracted_text = None
        self.messages = []
        self.messages_area.config(state='normal')
        self.messages_area.delete(1.0, tk.END)
        self.messages_area.config(state='disabled')
        self.camera_canvas.delete("all")
        self.status_label.config(text="Click 'Start Camera' to begin", fg=self.colors['text_secondary'])
        self.user_input.delete(0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    BookDiscussionApp(root)
    root.mainloop()