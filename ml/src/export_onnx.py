from optimum.exporters.onnx import main_export

def export():
    print("Local model ko ONNX format mein export kiya ja raha hai...")
    main_export(
        model_name_or_path="../models/xlmr-final",
        output="../models/onnx/",
        task="text-classification",
        opset=14
    )
    print("ONNX Export Successful! Files '../models/onnx/' mein save ho gayi hain.")

if __name__ == "__main__":
    export()