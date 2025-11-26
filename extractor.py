"""
Script 1: Process Exception PDFs → Document.json (Single Bucket)
"""
import json
import os
from datetime import datetime
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions

# ===== CONFIG - SINGLE BUCKET =====
PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us"
PROCESSOR_ID = "YOUR_EXTRACTOR_ID"
BUCKET_NAME = "your-single-bucket"  # One bucket for everything

# Subfolder paths within the bucket
EXCEPTION_PATH = "exceptions/"       # PDFs go here
OUTPUT_PATH = "document_jsons/"      # Generated JSONs go here

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    if '_' not in snake_str:
        return snake_str
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def ensure_camel_case(obj):
    """Recursively convert all dict keys to camelCase"""
    if isinstance(obj, dict):
        return {snake_to_camel(k): ensure_camel_case(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_camel_case(item) for item in obj]
    return obj

class ExtractorProcessor:
    def __init__(self):
        self.client_options = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
        self.processor_client = documentai.DocumentProcessorServiceClient(client_options=self.client_options)
        self.storage_client = storage.Client(project=PROJECT_ID)
        self.bucket = self.storage_client.bucket(BUCKET_NAME)
        self.processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
    
    def process_pdf(self, pdf_bytes: bytes) -> dict:
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
        )
        result = self.processor_client.process_document(request=request)
        document_dict = documentai.Document.to_dict(result.document)
        return ensure_camel_case(document_dict)
    
    def generate_doc_id(self, blob_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = os.path.splitext(os.path.basename(blob_name))[0]
        return f"{name}_{timestamp}"
    
    def save_json(self, data: dict, path: str):
        blob = self.bucket.blob(path)
        blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
        print(f"✓ gs://{BUCKET_NAME}/{path}")
    
    def run(self):
        print("="*60)
        print("Processing Exception PDFs")
        print("="*60)
        
        pdfs = [b for b in self.bucket.list_blobs(prefix=EXCEPTION_PATH) 
                if b.name.lower().endswith(".pdf")]
        
        if not pdfs:
            print(f"No PDFs found in {EXCEPTION_PATH}")
            return
        
        print(f"Found {len(pdfs)} PDFs\n")
        
        for i, blob in enumerate(pdfs, 1):
            try:
                print(f"[{i}/{len(pdfs)}] {blob.name}")
                doc_id = self.generate_doc_id(blob.name)
                print(f" ID: {doc_id}")
                
                pdf_bytes = blob.download_as_bytes()
                document_json = self.process_pdf(pdf_bytes)
                
                self.save_json(document_json, f"{OUTPUT_PATH}{doc_id}.json")
                print(f" Entities: {len(document_json.get('entities', []))}\n")
                
            except Exception as e:
                print(f" ERROR: {e}\n")

if __name__ == "__main__":
    ExtractorProcessor().run()


###############################################################################################################################################################################

"""
Script 2: Merge DB edits + prepare training data (Single Bucket)
"""
import json
import os
from google.cloud import storage

# ===== CONFIG - SINGLE BUCKET =====
PROJECT_ID = "YOUR_PROJECT_ID"
BUCKET_NAME = "your-single-bucket"

# Subfolder paths within the bucket
INPUT_PATH = "document_jsons/"      # From Script 1
EDITS_PATH = "human_edits/"         # DB corrections (optional)
TRAINING_PATH = "training_data/"    # Output for training

class EditMerger:
    def __init__(self):
        self.storage_client = storage.Client(project=PROJECT_ID)
        self.bucket = self.storage_client.bucket(BUCKET_NAME)
    
    def load_json(self, path: str) -> dict:
        blob = self.bucket.blob(path)
        if not blob.exists():
            raise FileNotFoundError(f"gs://{BUCKET_NAME}/{path}")
        return json.loads(blob.download_as_text())
    
    def get_db_edits(self, doc_id: str) -> dict:
        try:
            return self.load_json(f"{EDITS_PATH}{doc_id}_edits.json")
        except FileNotFoundError:
            return {}
    
    def apply_edits(self, document: dict, edits: dict) -> int:
        """Apply edits using camelCase fields"""
        count = 0
        for entity in document.get("entities", []):
            field = entity.get("type", "")
            if field in edits:
                entity["mentionText"] = edits[field]
                if "textAnchor" in entity:
                    entity["textAnchor"]["content"] = edits[field]
                entity["confidence"] = 1.0
                print(f"   ✓ {field} → {edits[field]}")
                count += 1
        return count
    
    def set_all_confidence_one(self, document: dict) -> int:
        count = 0
        for entity in document.get("entities", []):
            entity["confidence"] = 1.0
            count += 1
        return count
    
    def validate_entities(self, document: dict) -> list:
        errors = []
        for i, entity in enumerate(document.get("entities", [])):
            if not entity.get("type"):
                errors.append(f"Entity {i}: empty type")
            if "textAnchor" not in entity:
                errors.append(f"Entity {i}: missing textAnchor")
            if "mentionText" not in entity:
                errors.append(f"Entity {i}: missing mentionText")
        return errors
    
    def save_json(self, data: dict, path: str):
        blob = self.bucket.blob(path)
        blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
        print(f"   ✓ Saved to gs://{BUCKET_NAME}/{path}")
    
    def process(self, doc_id: str):
        print(f"\n{'='*60}")
        print(f"Document: {doc_id}")
        print('='*60)
        
        document = self.load_json(f"{INPUT_PATH}{doc_id}.json")
        print(f"   Entities: {len(document.get('entities', []))}")
        
        errors = self.validate_entities(document)
        if errors:
            print("   ❌ VALIDATION ERRORS:")
            for err in errors:
                print(f"      {err}")
            raise ValueError("Invalid entity structure")
        
        edits = self.get_db_edits(doc_id)
        if edits:
            print(f"   DB edits: {len(edits)}")
            count = self.apply_edits(document, edits)
            print(f"   Applied: {count}")
        else:
            print("   No edits - setting confidence=1.0")
            count = self.set_all_confidence_one(document)
            print(f"   Set confidence=1.0 for {count} entities")
        
        self.save_json(document, f"{TRAINING_PATH}{doc_id}_training.json")
    
    def run(self):
        print("="*60)
        print("Merging DB Edits + Preparing Training Data")
        print("="*60)
        
        blobs = self.bucket.list_blobs(prefix=INPUT_PATH)
        doc_ids = [
            os.path.basename(b.name).replace(".json", "")
            for b in blobs if b.name.endswith(".json")
        ]
        
        if not doc_ids:
            print(f"No documents found in {INPUT_PATH}")
            return
        
        print(f"Found {len(doc_ids)} documents\n")
        
        success = 0
        failed = 0
        
        for i, doc_id in enumerate(doc_ids, 1):
            try:
                print(f"[{i}/{len(doc_ids)}]")
                self.process(doc_id)
                success += 1
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Complete: {success} success, {failed} failed")
        print("="*60)

if __name__ == "__main__":
    EditMerger().run()


###################################################################################################################################################################################################################################################
"""
Script 3: Import training data to Document AI (Single Bucket)
"""
import time
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions

# ===== CONFIG - SINGLE BUCKET =====
PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us"
PROCESSOR_ID = "YOUR_EXTRACTOR_ID"
BUCKET_NAME = "your-single-bucket"

# Subfolder paths
TRAINING_PATH = "training_data/"

class TrainingImporter:
    def __init__(self):
        opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
        self.doc_service = documentai.DocumentServiceClient(client_options=opts)
        self.storage_client = storage.Client(project=PROJECT_ID)
        self.bucket = self.storage_client.bucket(BUCKET_NAME)
        
        processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
        self.dataset_name = f"{processor_name}/dataset"
    
    def import_to_training(self):
        print("="*60)
        print("Importing Training Data")
        print("="*60)
        
        ts = int(time.time() * 1000)
        import_path = f"training_data_import_{ts}/"
        import_prefix = f"gs://{BUCKET_NAME}/{import_path}"
        
        print(f"\nCopying to: {import_prefix}")
        src_blobs = list(self.bucket.list_blobs(prefix=TRAINING_PATH))
        
        json_count = 0
        for src in src_blobs:
            if src.name.endswith(".json"):
                dest_name = src.name.replace(TRAINING_PATH, import_path)
                self.bucket.copy_blob(src, self.bucket, dest_name)
                json_count += 1
        
        print(f"Copied {json_count} JSON files")
        
        if json_count == 0:
            print("❌ No training files found!")
            return
        
        batch_config = documentai.ImportDocumentsRequest.BatchDocumentsImportConfig(
            batch_input_config=documentai.BatchDocumentsInputConfig(
                gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=import_prefix)
            ),
            dataset_split=documentai.DatasetSplitType.DATASET_SPLIT_TRAIN
        )
        
        request = documentai.ImportDocumentsRequest(
            dataset=self.dataset_name,
            batch_documents_import_configs=[batch_config]
        )
        
        print("\nStarting import...")
        operation = self.doc_service.import_documents(request=request)
        
        ops_client = self.doc_service.transport.operations_client
        deadline = time.time() + 1800
        
        while time.time() < deadline:
            op = ops_client.get_operation(operation.operation.name)
            if op.done:
                if op.error and op.error.code != 0:
                    print(f"\n❌ Import failed: {op.error.message}")
                    raise RuntimeError(f"Import failed: {op.error.message}")
                print("\n✓ Import complete")
                break
            print(".", end="", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Import timeout")
        
        print("\nCleaning up temp files...")
        for blob in self.bucket.list_blobs(prefix=import_path):
            blob.delete()
        print("✓ Done")
    
    def run(self):
        print("="*60)
        print("Document AI Training Data Import")
        print("="*60)
        
        self.import_to_training()
        
        print("\n" + "="*60)
        print("Next: Go to Document AI Console → Train tab")
        print("="*60)

if __name__ == "__main__":
    TrainingImporter().run()
