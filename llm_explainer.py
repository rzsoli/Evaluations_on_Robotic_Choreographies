import gradio as gr
import json
import os
import random
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from typing import Optional

class SHAPInstanceExplorer:
    def __init__(self):
        self.base_path = Path("shap_sample_instances")
        self.target_values = [
            "Humancharacterization",
            "Humanreproducibility", 
            "Movementtechnique",
            "Publicinvolvement",
            "Rhythm",
            "Spaceuse",
            "Storytelling"
        ]
        self.gemini_api_key = ""  # To be filled by user
        self.gemini_model = None
        self.current_instance_data = None  # Store current instance for AI analysis
        self.shap_plots_path = Path("shap_sample_instances")  # Path to SHAP summary plots
    
    def get_random_instance(self, model_type, target_value):
        """Get a random JSON instance for the specified model type and target value"""
        folder_path = self.base_path / model_type / target_value
        
        if not folder_path.exists():
            return None, f"Error: Path {folder_path} does not exist"
        
        try:
            # Get all JSON files in the folder
            json_files = list(folder_path.glob("*.json"))
            
            if not json_files:
                return None, f"No JSON files found in {folder_path}"
            
            # Select a random file
            random_file = random.choice(json_files)
            
            # Load and return the JSON content
            with open(random_file, 'r') as f:
                instance_data = json.load(f)
            
            return instance_data, random_file.name
        
        except Exception as e:
            return None, f"Error loading instance: {str(e)}"
    
    def format_instance_display(self, instance_data, filename):
        """Format the instance data for display"""
        if instance_data is None:
            return "No data to display", "", ""
        
        # Format features as a table
        features_df = pd.DataFrame(list(instance_data['features'].items()), 
                                 columns=['Feature', 'Value'])
        
        # Format SHAP values as a table
        shap_df = pd.DataFrame(list(instance_data['shap_values'].items()), 
                             columns=['Feature', 'SHAP Value'])
        shap_df['SHAP Value'] = shap_df['SHAP Value'].round(6)
        
        # Sort SHAP values by absolute value (most important features first)
        shap_df['abs_shap'] = abs(shap_df['SHAP Value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False).drop('abs_shap', axis=1)
        
        # Create summary info
        summary = f"""
        **File:** {filename}
        **Instance ID:** {instance_data['instance_id']}
        **Prediction:** {instance_data['prediction']}
        """
        
        return summary, features_df, shap_df
    
    def configure_gemini_api(self, api_key: str):
        """Configure Gemini API with the provided key"""
        try:
            self.gemini_api_key = api_key.strip()
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                return "✅ Gemini API configured successfully!"
            else:
                return "⚠️ Please enter a valid API key"
        except Exception as e:
            return f"❌ Error configuring API: {str(e)}"
    
    def create_llm_prompt(self, instance_data, model_type, target_value):
        """Create a prompt template for LLM analysis"""
        features_text = "\n".join([f"- {k}: {v}" for k, v in instance_data['features'].items()])
        shap_text = "\n".join([f"- {k}: {v:.6f}" for k, v in instance_data['shap_values'].items()])
        
#         prompt = f"""
# You are an expert in machine learning explainability and robotic choreography evaluation. Please analyze the following SHAP explanation for a {model_type} model predicting {target_value.replace('_', ' ').title()}.

# **Instance Details:**
# - Instance ID: {instance_data['instance_id']}
# - Model Prediction: {instance_data['prediction']}
# - Model Type: {model_type.title()}
# - Target: {target_value.replace('_', ' ').title()}

# **Features:**
# {features_text}

# **SHAP Values (Feature Contributions):**
# {shap_text}

# Please provide a comprehensive analysis that includes:

# 1. **Key Insights**: What are the most important factors driving this prediction?
# 2. **Feature Analysis**: Explain the top 3-5 most influential features and their impact
# 3. **Model Behavior**: What does this tell us about how the model evaluates this aspect of choreography?
# 4. **Practical Implications**: What would a choreographer need to focus on to improve this evaluation aspect?
# 5. **Surprising Findings**: Are there any unexpected feature contributions?

# Please write your analysis in a clear, accessible way that would help both technical and non-technical users understand the model's decision-making process.
# """
        
        prompt = f"""
[SYSTEM]

You are an expert in machine learning explainability and Human-Robot Interaction (HRI). Your task is to analyze a single prediction from a machine learning model that evaluates robot choreography and generate a comprehensive, five-part analysis.

Your explanation must be clear, insightful, and accessible to a dual audience:
1.  **Non-technical users** (e.g., choreographers, designers).
2.  **Technical users** (e.g., HRI engineers, data scientists).

You will be given the specific feature values for a choreography and the corresponding SHAP values, which quantify how much each feature contributed to the model's final prediction. A positive SHAP value pushed the prediction higher, while a negative value pushed it lower.

**General Context from the Full Report:**
Remember that across most models, the three most influential features are consistently **`nMovements`** (number of movements), **`timeDuration`**, and **`musicBPM`** (music tempo). Your analysis should pay special attention to these, but also highlight other interesting contributors.

**Required Output Structure:**
Please structure your analysis using exactly these five sections:
1.  **Key Insights**: A one-sentence summary of the most important factors driving this specific prediction.
2.  **Feature Analysis**: Explain the impact of the top 3-5 most influential features (both positive and negative).
3.  **Model Behavior**: Generalize from this single instance to describe what the model seems to have learned about this evaluation metric.
4.  **Practical Implications**: Provide actionable advice for a choreographer or designer looking to improve a robot's score for this target.
5.  **Surprising Findings**: Point out any unexpected or subtle feature contributions that are not immediately obvious.

---
### **Examples**

<example>
<input>
You are an expert in machine learning explainability and robotic choreography evaluation. Please analyze the following SHAP explanation for a **classification** model predicting **Human Characterization**.

**Instance Details:**
- Instance ID: 62
- Model Prediction: 1.0 (High Characterization)
- Model Type: Classification
- Target: Human Characterization

**Features:**
- timeDuration: 125
- nMovements: 15
- musicBPM: 152
- movementsDifficulty: 3
- humanMovements: 2

**SHAP Values (Feature Contributions):**
- timeDuration: 0.095896
- musicBPM: 0.067902
- nMovements: 0.036426
- movementsTransitionsDuration: 0.014255
- handsMovement: -0.003240
</input>
<analysis>
Here is the analysis of the model's prediction for instance 62 on **Human Characterization**.

**1. Key Insights**
The model predicted a high `Human Characterization` score primarily because the choreography has a long duration and is set to fast-paced music, creating a sense of energy and presence.

**2. Feature Analysis**
*   **timeDuration (Value: 125s, SHAP: +0.096):** This was the single most positive contributor. The long duration gives the robot more opportunity to perform and develop a character, which the model has learned to associate strongly with a "human-like" quality.
*   **musicBPM (Value: 152, SHAP: +0.068):** The high tempo music was the second most important factor. The model interprets this energy and dynamism as a key component of human-like expression, pushing the score significantly higher.
*   **nMovements (Value: 15, SHAP: +0.036):** Having a relatively high number of movements also contributed positively. This suggests that variety and complexity in the routine are seen by the model as indicators of human-like performance.
*   **handsMovement (Value: 1, SHAP: -0.003):** Although a small effect, it's worth noting that the lowest level of hand movement had a slight negative impact. This hints that the model may penalize routines that lack fine motor detail in the hands.

**3. Model Behavior**
This instance shows that the model has learned to equate `Human Characterization` with dynamic, sustained, and varied performances. It prioritizes overall energy (high BPM) and stage presence (long duration) as primary indicators, suggesting it doesn't just count movements but also considers the context in which they are performed.

**4. Practical Implications**
To consistently score high on `Human Characterization`, a choreographer should design routines that are:
*   **Longer:** Avoid very short performances; aim for routines over 2 minutes.
*   **Energetic:** Select music with a higher tempo (BPM).
*   **Varied:** Incorporate a healthy number of different movements.
*   **Detailed:** Pay attention to expressive elements like hand and arm movements, as the model may be sensitive to a lack of detail.

**5. Surprising Findings**
It's interesting that `movementsDifficulty` (level 3) had almost no impact on the prediction (SHAP: +0.00001). For this specific choreography, the model decided that the performance's duration and energy were far more important for `Human Characterization` than the technical difficulty of the moves themselves.
</analysis>
</example>

<example>
<input>
You are an expert in machine learning explainability and robotic choreography evaluation. Please analyze the following SHAP explanation for a **regressor** model predicting **Storytelling**.

**Instance Details:**
- Instance ID: 9
- Model Prediction: 2.58 (Low Storytelling Score)
- Model Type: Regressor
- Target: Storytelling

**Features:**
- nMovements: 19
- musicBPM: 87
- movementsTransitionsDuration: 2
- balance: 1
- handsMovement: 3

**SHAP Values (Feature Contributions):**
- nMovements: -0.290114
- musicBPM: -0.079240
- balance: -0.022600
- movementsTransitionsDuration: 0.043546
- handsMovement: 0.025451
</input>
<analysis>
Here is the analysis of the model's prediction for instance 9 on **Storytelling**.

**1. Key Insights**
The model predicted a low `Storytelling` score of 2.58 primarily because it found the high number of movements cluttered the narrative, and the slow music lacked the dynamic drive for compelling storytelling.

**2. Feature Analysis**
*   **nMovements (Value: 19, SHAP: -0.290):** This was by far the most significant factor, heavily penalizing the score. The model has learned that too many distinct movements in a routine can make it feel disjointed and prevent a clear story from emerging.
*   **musicBPM (Value: 87, SHAP: -0.079):** The slow tempo was the second largest negative contributor. This suggests the model associates effective storytelling with a certain level of energy or pacing, which was lacking here.
*   **movementsTransitionsDuration (Value: 2, SHAP: +0.044):** This was the most significant positive feature. A medium transition duration helped salvage some of the score, indicating that even in a cluttered routine, the model values well-paced connections between movements.
*   **balance (Value: 1, SHAP: -0.023):** A low level of balance movements also detracted from the score. The model may have learned that moments of controlled balance contribute to a performance's narrative focus and composure.

**3. Model Behavior**
For `Storytelling`, the model seems to operate on a "less is more" principle. It clearly penalizes choreographies that it deems overly complex or "busy" (`high nMovements`). It also indicates that musicality is key, associating higher-energy music with more engaging narratives. The model doesn't just look for movement; it looks for *coherent and focused* movement.

**4. Practical Implications**
To improve a robot's `Storytelling` score, a choreographer should:
*   **Simplify:** Drastically reduce the number of distinct movements to focus on a core set that clearly tells a story.
*   **Be Deliberate with Music:** Choose music with a more dynamic tempo that supports the narrative arc. Avoid slow, monotonous tracks.
*   **Focus on Flow:** Ensure smooth and meaningful transitions between movements, as the model rewards this.
*   **Incorporate Poise:** Include moments of balance or control to add narrative weight.

**5. Surprising Findings**
It is quite surprising that `timeDuration` (121s) had a very small positive impact, despite being a globally important feature. For this instance, the negative effect of having too many movements completely overshadowed the benefit of having a longer performance time. This demonstrates a clear feature interaction: a long duration is only good for storytelling if it's not filled with chaotic movement.
</analysis>
</example>

---
### **Your Turn: Analyze the Following Instance**
You are an expert in machine learning explainability and robotic choreography evaluation. Please analyze the following SHAP explanation for a {model_type} model predicting {target_value.replace('_', ' ').title()}.

**Instance Details:**
- Instance ID: {instance_data['instance_id']}
- Model Prediction: {instance_data['prediction']}
- Model Type: {model_type.title()}
- Target: {target_value.replace('_', ' ').title()}

**Features:**
{features_text}

**SHAP Values (Feature Contributions):**
{shap_text}

Now, please provide your comprehensive five-part analysis for the instance defined above.
"""
        return prompt
    
    def get_shap_summary_plot(self, target_value):
        """Get the SHAP summary plot for the specified target value"""
        try:
            plot_filename = f"{target_value}_shap_summary.png"
            plot_path = self.shap_plots_path / plot_filename
            
            if plot_path.exists():
                return str(plot_path)
            else:
                return None
        except Exception as e:
            print(f"Error loading SHAP plot: {e}")
            return None
    
    def load_instance(self, model_type, target_value):
        """Load a random instance and format it for display"""
        instance_data, filename = self.get_random_instance(model_type, target_value)
        
        if instance_data is None:
            self.current_instance_data = None
            shap_plot = self.get_shap_summary_plot(target_value)
            return filename, pd.DataFrame(), pd.DataFrame(), "No instance data available", gr.update(interactive=False), shap_plot
        
        # Store current instance data for potential AI analysis
        self.current_instance_data = {
            'data': instance_data,
            'model_type': model_type,
            'target_value': target_value
        }
        
        summary, features_df, shap_df = self.format_instance_display(instance_data, filename)
        
        # Get SHAP summary plot for this target value
        shap_plot = self.get_shap_summary_plot(target_value)
        
        # Clear previous AI analysis and enable Ask AI button
        initial_ai_message = "Instance loaded! Click **Ask AI** below to get detailed analysis."
        
        return summary, features_df, shap_df, initial_ai_message, gr.update(interactive=True), shap_plot
    
    def ask_ai_analysis(self):
        """Generate AI analysis for the currently loaded instance"""
        if not self.current_instance_data:
            return '⚠️ **Error:** No instance loaded. Please load an instance first.'
        
        if not self.gemini_model or not self.gemini_api_key:
            return '⚠️ **Error:** Please configure Gemini API key first.'
        
        try:
            # Generate analysis using stored instance data
            instance_data = self.current_instance_data['data']
            model_type = self.current_instance_data['model_type']
            target_value = self.current_instance_data['target_value']
            
            prompt = self.create_llm_prompt(instance_data, model_type, target_value)
            print(prompt)
            response = self.gemini_model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            return f'❌ **Error generating AI analysis:** {str(e)}'

def create_gradio_interface():
    explorer = SHAPInstanceExplorer()
    
    # Custom CSS for better styling and readability
    custom_css = """
    .llm-output {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        max-height: 500px;
        overflow-y: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        font-size: 14px;
    }
    .llm-output h1, .llm-output h2, .llm-output h3, .llm-output h4, .llm-output h5, .llm-output h6 {
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .llm-output p {
        margin-bottom: 12px;
        color: #2c3e50;
    }
    .llm-output strong {
        color: #2980b9;
        font-weight: 600;
    }
    .llm-output ul, .llm-output ol {
        margin-left: 20px;
        margin-bottom: 12px;
    }
    .llm-output li {
        margin-bottom: 6px;
        color: #2c3e50;
    }
    .llm-output code {
        background-color: #f1f2f6;
        color: #e74c3c;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
    }
    .llm-output blockquote {
        border-left: 4px solid #3498db;
        padding-left: 15px;
        margin: 15px 0;
        background-color: #ecf0f1;
        font-style: italic;
    }
    .api-status {
        font-weight: bold;
    }
    .ask-ai-button {
        margin-left: auto !important;
        background-color: #5865f2 !important;
        border-color: #5865f2 !important;
    }
    .ask-ai-button:hover {
        background-color: #4752c4 !important;
    }
    .ask-ai-button:disabled {
        background-color: #40444b !important;
        border-color: #40444b !important;
        color: #72767d !important;
    }
    """
    
    with gr.Blocks(title="SHAP Instance Explorer", theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("# 🔍 SHAP Instance Explorer")
        gr.Markdown("Explore SHAP explanations for robotic choreography evaluation models")
        
        # API Configuration Section
        with gr.Accordion("🔧 Gemini API Configuration", open=True):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Enter your Gemini API key here...",
                    type="password",  # Secure password field
                    scale=3
                )
                configure_btn = gr.Button(
                    "Configure API",
                    variant="secondary",
                    scale=1
                )
                api_status = gr.Markdown("⚪ API not configured")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model type selection
                model_type = gr.Radio(
                    choices=["classification", "regression"],
                    value="classification",
                    label="Model Type",
                    info="Choose between classification or regression models"
                )
                
                # Target value selection
                target_value = gr.Dropdown(
                    choices=explorer.target_values,
                    value=explorer.target_values[0],
                    label="Target Value",
                    info="Select the choreography evaluation aspect"
                )
                
                # Load button
                load_btn = gr.Button(
                    "🎲 Load Random Instance", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Summary display
                summary_display = gr.Markdown("Click 'Load Random Instance' to begin exploring!")
                
                # LLM Analysis Section
                with gr.Row():
                    gr.Markdown("### 🤖 AI Analysis")
                    ask_ai_btn = gr.Button(
                        "🤖 Ask AI",
                        variant="primary",
                        size="sm",
                        interactive=False,
                        scale=0,
                        elem_classes=["ask-ai-button"]
                    )
                
                llm_output = gr.Markdown(
                    "Load an instance first, then click \"Ask AI\" to get detailed analysis...",
                    elem_classes=["llm-output"]
                )
        
        # Data tables and SHAP plot
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Features")
                features_table = gr.Dataframe(
                    headers=["Feature", "Value"],
                    datatype=["str", "number"],
                    interactive=False,
                    wrap=True
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 SHAP Values (Ranked by Importance)")
                shap_table = gr.Dataframe(
                    headers=["Feature", "SHAP Value"],
                    datatype=["str", "number"],
                    interactive=False,
                    wrap=True
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 SHAP Summary Plot")
                shap_plot_display = gr.Image(
                    label="Feature Importance Overview",
                    show_label=False,
                    container=True,
                    height=400,
                    value=explorer.get_shap_summary_plot(explorer.target_values[0])
                )
        
        # Info section
        with gr.Accordion("ℹ️ About SHAP Values", open=False):
            gr.Markdown("""
            **SHAP (SHapley Additive exPlanations)** values explain the contribution of each feature to the model's prediction:
            
            - **Positive values**: Features that increase the prediction
            - **Negative values**: Features that decrease the prediction
            - **Magnitude**: Shows how much the feature contributes
            - **Zero**: Features with no contribution to this specific prediction
            
            Features are ranked by absolute SHAP value (most important first).
            """)
        
        # Set up the interactions
        configure_btn.click(
            fn=explorer.configure_gemini_api,  # Calls the API setup function
            inputs=[api_key_input],            # Takes the API key
            outputs=[api_status]               # Shows configuration status
        )
        
        load_btn.click(
            fn=explorer.load_instance,
            inputs=[model_type, target_value],
            outputs=[summary_display, features_table, shap_table, llm_output, ask_ai_btn, shap_plot_display]
        )
        
        ask_ai_btn.click(
            fn=explorer.ask_ai_analysis,
            inputs=[],
            outputs=[llm_output]
        )
        
        # Update SHAP plot when target value changes
        target_value.change(
            fn=explorer.get_shap_summary_plot,
            inputs=[target_value],
            outputs=[shap_plot_display]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
