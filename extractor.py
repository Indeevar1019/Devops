"""
Script 1: Process Exception PDFs → Document.json (with camelCase guarantee)
"""
import json
import os
from datetime import datetime
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions

# ===== CONFIG =====
PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us"
PROCESSOR_ID = "YOUR_EXTRACTOR_ID"
EXCEPTION_BUCKET = "your-exception-bucket"
OUTPUT_BUCKET = "your-output-bucket"

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
        self.processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
    
    def process_pdf(self, pdf_bytes: bytes) -> dict:
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
        )
        result = self.processor_client.process_document(request=request)
        
        # Convert to dict and ensure camelCase
        document_dict = documentai.Document.to_dict(result.document)
        return ensure_camel_case(document_dict)
    
    def generate_doc_id(self, blob_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = os.path.splitext(os.path.basename(blob_name))[0]
        return f"{name}_{timestamp}"
    
    def save_json(self, data: dict, bucket: str, path: str):
        blob = self.storage_client.bucket(bucket).blob(path)
        blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
        print(f"✓ gs://{bucket}/{path}")
    
    def run(self):
        print("="*60)
        print("Processing Exception PDFs")
        print("="*60)
        
        bucket = self.storage_client.bucket(EXCEPTION_BUCKET)
        pdfs = [b for b in bucket.list_blobs() if b.name.lower().endswith(".pdf")]
        
        if not pdfs:
            print("No PDFs found")
            return
        
        print(f"Found {len(pdfs)} PDFs\n")
        
        for i, blob in enumerate(pdfs, 1):
            try:
                print(f"[{i}/{len(pdfs)}] {blob.name}")
                doc_id = self.generate_doc_id(blob.name)
                print(f" ID: {doc_id}")
                
                pdf_bytes = blob.download_as_bytes()
                document_json = self.process_pdf(pdf_bytes)
                
                self.save_json(document_json, OUTPUT_BUCKET, f"document_jsons/{doc_id}.json")
                print(f" Entities: {len(document_json.get('entities', []))}\n")
                
            except Exception as e:
                print(f" ERROR: {e}\n")

if __name__ == "__main__":
    ExtractorProcessor().run()

##################################################################################################################################################################################################################################################
"""
Script 2: Merge DB edits + prepare training data (FIXED - camelCase)
"""
import json
import os
from google.cloud import storage

# ===== CONFIG =====
PROJECT_ID = "YOUR_PROJECT_ID"
INPUT_BUCKET = "your-output-bucket"
TRAINING_BUCKET = "your-training-bucket"

class EditMerger:
    def __init__(self):
        self.storage_client = storage.Client(project=PROJECT_ID)
    
    def load_json(self, bucket: str, path: str) -> dict:
        blob = self.storage_client.bucket(bucket).blob(path)
        if not blob.exists():
            raise FileNotFoundError(f"gs://{bucket}/{path}")
        return json.loads(blob.download_as_text())
    
    def get_db_edits(self, doc_id: str) -> dict:
        try:
            return self.load_json(INPUT_BUCKET, f"human_edits/{doc_id}_edits.json")
        except FileNotFoundError:
            return {}
    
    def apply_edits(self, document: dict, edits: dict) -> int:
        """Apply edits using CORRECT camelCase fields"""
        count = 0
        for entity in document.get("entities", []):
            field = entity.get("type", "")
            if field in edits:
                # ✅ Use camelCase
                entity["mentionText"] = edits[field]
                
                if "textAnchor" in entity:
                    entity["textAnchor"]["content"] = edits[field]
                
                entity["confidence"] = 1.0
                print(f"   ✓ {field} → {edits[field]}")
                count += 1
        return count
    
    def set_all_confidence_one(self, document: dict) -> int:
        """Set all entities to confidence=1.0"""
        count = 0
        for entity in document.get("entities", []):
            entity["confidence"] = 1.0
            count += 1
        return count
    
    def validate_entities(self, document: dict) -> list:
        """Validate entities have correct camelCase structure"""
        errors = []
        for i, entity in enumerate(document.get("entities", [])):
            if not entity.get("type"):
                errors.append(f"Entity {i}: empty type")
            if "textAnchor" not in entity:
                errors.append(f"Entity {i}: missing textAnchor")
            if "mentionText" not in entity:
                errors.append(f"Entity {i}: missing mentionText")
        return errors
    
    def save_json(self, data: dict, bucket: str, path: str):
        blob = self.storage_client.bucket(bucket).blob(path)
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type="application/json"
        )
        print(f"   ✓ Saved to gs://{bucket}/{path}")
    
    def process(self, doc_id: str):
        print(f"\n{'='*60}")
        print(f"Document: {doc_id}")
        print('='*60)
        
        # Load document JSON (from Script 1)
        document = self.load_json(INPUT_BUCKET, f"document_jsons/{doc_id}.json")
        print(f"   Entities: {len(document.get('entities', []))}")
        
        # Validate structure
        errors = self.validate_entities(document)
        if errors:
            print("   ❌ VALIDATION ERRORS:")
            for err in errors:
                print(f"      {err}")
            raise ValueError("Invalid entity structure")
        
        # Apply DB edits if available
        edits = self.get_db_edits(doc_id)
        if edits:
            print(f"   DB edits: {len(edits)}")
            count = self.apply_edits(document, edits)
            print(f"   Applied: {count}")
        else:
            print("   No edits - setting confidence=1.0")
            count = self.set_all_confidence_one(document)
            print(f"   Set confidence=1.0 for {count} entities")
        
        # Save training JSON
        self.save_json(document, TRAINING_BUCKET, f"training_data/{doc_id}_training.json")
    
    def run(self):
        print("="*60)
        print("Merging DB Edits + Preparing Training Data")
        print("="*60)
        
        bucket = self.storage_client.bucket(INPUT_BUCKET)
        blobs = bucket.list_blobs(prefix="document_jsons/")
        doc_ids = [
            os.path.basename(b.name).replace(".json", "")
            for b in blobs if b.name.endswith(".json")
        ]
        
        if not doc_ids:
            print("No documents found")
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

#############################################################################################################################################################################################################################################

"""
Script 3: Import training data to Document AI (NO dataset clearing)
"""
import time
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions

# ===== CONFIG =====
PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us"
PROCESSOR_ID = "YOUR_EXTRACTOR_ID"
TRAINING_BUCKET = "your-training-bucket"

class TrainingImporter:
    def __init__(self):
        opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
        self.doc_service = documentai.DocumentServiceClient(client_options=opts)
        self.storage_client = storage.Client(project=PROJECT_ID)
        
        processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
        self.dataset_name = f"{processor_name}/dataset"
    
    def import_to_training(self):
        print("="*60)
        print("Importing Training Data")
        print("="*60)
        
        # Create timestamped import location
        ts = int(time.time() * 1000)
        import_prefix = f"gs://{TRAINING_BUCKET}/training_data_import_{ts}/"
        
        print(f"\nCopying to: {import_prefix}")
        bucket = self.storage_client.bucket(TRAINING_BUCKET)
        src_blobs = list(bucket.list_blobs(prefix="training_data/"))
        
        json_count = 0
        for src in src_blobs:
            if src.name.endswith(".json"):
                dest_name = src.name.replace("training_data/", f"training_data_import_{ts}/")
                bucket.copy_blob(src, bucket, dest_name)
                json_count += 1
        
        print(f"Copied {json_count} JSON files")
        
        if json_count == 0:
            print("❌ No training files found!")
            return
        
        # Import to dataset (TRAIN only)
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
        
        # Wait for completion
        ops_client = self.doc_service.transport.operations_client
        deadline = time.time() + 1800  # 30 min timeout
        
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
            raise TimeoutError("Import timeout after 30 minutes")
        
        # Cleanup temp import files
        print("\nCleaning up temp files...")
        for blob in bucket.list_blobs(prefix=f"training_data_import_{ts}/"):
            blob.delete()
        print("✓ Cleanup done")
    
    def run(self):
        print("="*60)
        print("Document AI Training Data Import")
        print("="*60)
        
        self.import_to_training()
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("1. Go to Document AI Console")
        print("2. Navigate to your processor's 'Train' tab")
        print("3. Review imported documents")
        print("4. Click 'Train New Version'")
        print("="*60)

if __name__ == "__main__":
    TrainingImporter().run()
