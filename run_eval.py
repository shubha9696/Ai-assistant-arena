import os
import json
from evaluation.report_generator import ReportGenerator

def main():
    print("Generating evaluation reports from benchmark dataset...")
    
    # Load pre-computed results
    results_path = "evaluation/precomputed_results.json"
    if not os.path.exists(results_path):
        print(f"Error: Pre-computed results not found at {results_path}")
        return
        
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    oss_results = data["oss"]
    frontier_results = data["frontier"]
    
    # Set model names
    oss_model = "Qwen/Qwen2.5-7B-Instruct"
    frontier_model = "gemini-1.5-flash"
    
    # Output paths
    os.makedirs("artifacts", exist_ok=True)
    pdf_path = "artifacts/evaluation_report.pdf"
    md_path = "artifacts/evaluation_report.md"
    
    # Generate reports
    ReportGenerator.generate_markdown_report(oss_results, frontier_results, oss_model, frontier_model, md_path)
    print(f"Generated Markdown report at: {md_path}")
    
    ReportGenerator.generate_pdf_report(oss_results, frontier_results, oss_model, frontier_model, pdf_path)
    print(f"Generated PDF report at: {pdf_path}")
    
    print("Success! Reports generated successfully.")

if __name__ == "__main__":
    main()
