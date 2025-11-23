import os
import io
import tempfile
import time
from pathlib import Path
from collections import Counter

import streamlit as st
from PIL import Image
import numpy as np

# ============================================
#        CONFIG GLOBAL & DESIGN
# ============================================
st.set_page_config(
    page_title="Détecteur de Poubelle",
    layout="centered",
)

# ---- CSS custom (fond blanc + style moderne)
st.markdown("""
<style>
body {
    background-color: white;
    color: #222;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    color: #0A3D62;
}
.sidebar .sidebar-content {
    background-color: #F5F6FA;
}
.stButton > button {
    background-color: #0984E3;
    color: white;
    padding: 8px 18px;
    border-radius: 6px;
}
.stButton > button:hover {
    background-color: #74B9FF;
}
</style>
""", unsafe_allow_html=True)

# ============================================
#     LOAD YOLO OR FAIL
# ============================================
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    st.error("La librairie 'ultralytics' est requise : `pip install ultralytics`")
    ULTRALYTICS_AVAILABLE = False

MODEL_LOCAL = Path("best.pt")

if "model" not in st.session_state:
    st.session_state.model = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None


@st.cache_resource(show_spinner=False)
def load_model(path):
    if not ULTRALYTICS_AVAILABLE:
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None


def run_image_inference(model, img_source, conf=0.25, iou=0.45):
    """
    Effectue l'inférence sur une image
    img_source peut être un chemin ou un array numpy
    """
    try:
        results = model.predict(source=img_source, conf=conf, iou=iou, save=False)

        if not results:
            return None, None, "Aucun résultat"

        r = results[0]
        annotated = r.plot()

        boxes = []
        try:
            xyxy = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            classes = r.boxes.cls.tolist()
            names = r.names if hasattr(r, "names") else model.names

            for b, c, cl in zip(xyxy, confs, classes):
                boxes.append({
                    "xmin": int(b[0]), "ymin": int(b[1]),
                    "xmax": int(b[2]), "ymax": int(b[3]),
                    "conf": float(c),
                    "class": names.get(int(cl), str(cl))
                })
        except:
            pass

        if boxes:
            detected_labels = [b["class"].lower() for b in boxes]
            if any(k in d for d in detected_labels for k in ["plein", "full"]):
                status = "Poubelle pleine"
            elif any(k in d for d in detected_labels for k in ["vide", "empty"]):
                status = "Poubelle vide"
            else:
                status = "Poubelle détectée mais état incertain"
        else:
            status = "Aucune poubelle détectée"

        return annotated, boxes, status

    except Exception as e:
        st.error(f"Erreur d'inférence : {e}")
        return None, None, "Erreur"


def process_video_with_pil(model, video_file, conf=0.25, iou=0.45, frame_skip=5, resize_factor=0.5):
    """
    Traite une vidéo en utilisant PIL plutôt que cv2
    Note: Moins performant que cv2 mais évite les problèmes de dépendances
    """
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False
        st.warning("⚠️ CV2 non disponible. Traitement vidéo limité aux images extraites.")
        return
    
    # Si cv2 est disponible, on l'utilise quand même pour la vidéo (plus efficace)
    if not CV2_AVAILABLE:
        st.error("Le traitement vidéo nécessite opencv-python-headless. Utilisez l'onglet Image à la place.")
        return
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la vidéo")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    
    frame_count = 0
    processed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Redimensionner avec cv2
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convertir BGR vers RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Traiter avec YOLO
            results = model.predict(source=frame_rgb, conf=conf, iou=iou, save=False, verbose=False)
            
            if results:
                r = results[0]
                annotated = r.plot()
                
                boxes = []
                try:
                    xyxy = r.boxes.xyxy.tolist()
                    confs = r.boxes.conf.tolist()
                    classes = r.boxes.cls.tolist()
                    names = r.names if hasattr(r, "names") else model.names

                    for b, c, cl in zip(xyxy, confs, classes):
                        boxes.append({
                            "class": names.get(int(cl), str(cl)),
                            "conf": float(c)
                        })
                except:
                    pass
                
                # Calculer le statut
                status = "Aucune détection"
                if boxes:
                    detected_labels = [b["class"].lower() for b in boxes]
                    if any(k in d for d in detected_labels for k in ["plein", "full"]):
                        status = "🔴 Poubelle pleine"
                    elif any(k in d for d in detected_labels for k in ["vide", "empty"]):
                        status = "🟢 Poubelle vide"
                    else:
                        status = "🟡 Poubelle détectée"
                
                progress = frame_count / total_frames
                processed_count += 1
                
                yield annotated, status, boxes, progress, frame_count, total_frames, processed_count
        
        frame_count += 1
    
    cap.release()


def cleanup(path):
    try:
        if Path(path).exists():
            os.remove(path)
    except:
        pass


def read_file_bytes(path):
    """Renvoie le contenu binaire du fichier si disponible, sinon None."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


# ============================================
#          INTERFACE STREAMLIT
# ============================================
def main():

    st.title("🗑 Détection automatique de poubelles")

    # -------- SIDEBAR ---------
    st.sidebar.header("⚙ Paramètres")

    src = st.sidebar.radio(
        "Source du modèle",
        ["Modèle local", "Uploader un .pt"]
    )

    uploaded_model = None
    if src == "Uploader un .pt":
        uploaded_model = st.sidebar.file_uploader("Uploader un .pt", type=["pt"])

    conf = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.25)
    iou = st.sidebar.slider("Seuil NMS (IoU)", 0.0, 1.0, 0.45)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Options vidéo")
    frame_skip = st.sidebar.slider(
        "Traiter 1 frame sur", 
        1, 30, 5,
        help="Augmenter pour un traitement plus rapide (ex: 10 = 10x plus rapide)"
    )
    
    resize_factor = st.sidebar.slider(
        "Résolution vidéo",
        0.25, 1.0, 0.5, 0.25,
        help="Réduire pour accélérer (0.5 = 4x plus rapide)"
    )
    
    display_mode = st.sidebar.radio(
        "Mode d'affichage",
        ["Streaming en temps réel", "Traitement puis résultat final"],
        help="Le mode final est plus rapide"
    )

    if st.sidebar.button("Recharger le modèle"):
        load_model.clear()
        st.session_state.model = None
        st.session_state.model_path = None
        st.experimental_rerun()

    # Bouton pour télécharger le modèle actuel (si présent)
    model_download_path = None
    if st.session_state.model_path:
        model_download_path = st.session_state.model_path
    else:
        if MODEL_LOCAL.exists():
            model_download_path = str(MODEL_LOCAL)

    if model_download_path and Path(model_download_path).exists():
        model_bytes = read_file_bytes(model_download_path)
        if model_bytes is not None:
            st.sidebar.download_button(
                "Télécharger le modèle (.pt)",
                data=model_bytes,
                file_name=Path(model_download_path).name,
                mime="application/octet-stream",
            )
        else:
            st.sidebar.info("Impossible de lire le fichier du modèle pour le téléchargement.")

    # ------- Load Model --------
    new_path = None

    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as t:
            t.write(uploaded_model.read())
            new_path = t.name
        st.sidebar.success("Modèle chargé depuis upload.")
    else:
        if MODEL_LOCAL.exists():
            new_path = str(MODEL_LOCAL)
            st.sidebar.info(f"Utilisation du modèle local : {MODEL_LOCAL.name}")
        else:
            st.sidebar.warning("Aucun modèle disponible.")

    if new_path and st.session_state.model_path != new_path:
        with st.spinner("Chargement du modèle..."):
            st.session_state.model = load_model(new_path)
            st.session_state.model_path = new_path

    model = st.session_state.model

    # ============================================
    # TABS POUR IMAGE, VIDEO ET WEBCAM
    # ============================================
    tab1, tab2, tab3 = st.tabs(["📷 Image", "🎥 Vidéo", "📹 Webcam"])
    
    # ============================================
    # TAB 1: IMAGE (avec PIL uniquement)
    # ============================================
    with tab1:
        st.header("📷 Test sur une image")
        uploaded_image = st.file_uploader(
            "Uploader une image",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )

        # Galerie d'images déjà testées par d'autres
        samples_dir = Path("tested_images")
        sample_nparray = None
        if samples_dir.exists() and any(samples_dir.iterdir()):
            st.markdown("---")
            st.subheader("📚 Images déjà testées")
            img_paths = [p for p in samples_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            if img_paths:
                cols = st.columns(4)
                for i, p in enumerate(img_paths):
                    col = cols[i % 4]
                    try:
                        col.image(str(p), use_column_width=True)
                    except Exception:
                        col.write(p.name)
                    if col.button("Tester", key=f"test_sample_{i}"):
                        st.session_state.sample_to_test = str(p)
                        st.experimental_rerun()
            else:
                st.info("Aucune image valide dans `tested_images`. Ajoutez des .jpg/.png.")

        # Si l'utilisateur a choisi une image sample, on la charge
        if "sample_to_test" in st.session_state and st.session_state.sample_to_test:
            try:
                sample_path = st.session_state.sample_to_test
                sample_img = Image.open(sample_path)
                uploaded_image = None
                st.markdown(f"**Image sélectionnée :** {Path(sample_path).name}")
                st.image(sample_img, use_container_width=True)
                sample_nparray = np.array(sample_img)
            except Exception as e:
                st.error(f"Impossible de charger l'image sample : {e}")
                sample_nparray = None

        if (uploaded_image or sample_nparray is not None) and model:
            # Utiliser PIL pour charger l'image
            if uploaded_image:
                img = Image.open(uploaded_image)
                img_array = np.array(img)
            else:
                img_array = sample_nparray
            
            with st.spinner("Analyse en cours..."):
                # Passer directement l'array numpy à YOLO
                annotated, boxes, status = run_image_inference(
                    model, img_array, conf, iou
                )

            if annotated is not None:
                st.subheader(f"Résultat : {status}")
                st.image(annotated, use_container_width=True)

                if boxes:
                    st.markdown("### 📋 Détections détectées")
                    st.dataframe(boxes, use_container_width=True)

                buf = io.BytesIO()
                Image.fromarray(annotated.astype("uint8")).save(buf, format="JPEG")
                buf.seek(0)
                st.download_button(
                    "📥 Télécharger l'image annotée",
                    buf,
                    "resultat.jpg",
                    mime="image/jpeg"
                )

            else:
                st.warning("Aucun résultat. Vérifiez l'image et les paramètres.")

        elif (uploaded_image or ("sample_to_test" in st.session_state and st.session_state.sample_to_test)) and not model:
            st.error("Veuillez charger le modèle d'abord.")
    
    # ============================================
    # TAB 2: VIDEO (nécessite cv2 mais géré gracieusement)
    # ============================================
    with tab2:
        st.header("🎥 Test sur une vidéo")
        
        # Vérifier si cv2 est disponible
        try:
            import cv2
            CV2_AVAILABLE = True
        except ImportError:
            CV2_AVAILABLE = False
        
        if not CV2_AVAILABLE:
            st.warning("""
            ⚠️ **Le traitement vidéo nécessite OpenCV (cv2)**
            
            Pour activer cette fonctionnalité :
            1. Ajoutez `opencv-python-headless` dans requirements.txt
            2. Créez un fichier `packages.txt` avec :
               ```
               libgl1
               libglib2.0-0
               ```
            3. Redéployez l'application
            
            En attendant, vous pouvez extraire une frame de votre vidéo et l'utiliser dans l'onglet "📷 Image".
            """)
        else:
            uploaded_video = st.file_uploader(
                "Uploader une vidéo",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader"
            )

            if uploaded_video and model:
                # Sauvegarder la vidéo temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
                    t.write(uploaded_video.read())
                    video_path = t.name

                st.success("✅ Vidéo chargée avec succès!")
                
                if st.button("▶️ Lancer l'analyse vidéo"):
                    
                    if display_mode == "Streaming en temps réel":
                        # MODE STREAMING
                        status_placeholder = st.empty()
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        info_placeholder = st.empty()
                        
                        all_detections = []
                        
                        for annotated, status, boxes, progress, frame_num, total, processed in process_video_with_pil(
                            model, video_path, conf, iou, frame_skip, resize_factor
                        ):
                            status_placeholder.markdown(f"### {status}")
                            video_placeholder.image(annotated, use_container_width=True)
                            progress_bar.progress(progress)
                            info_placeholder.text(f"Frame {frame_num}/{total} (traitées: {processed})")
                            
                            if boxes:
                                all_detections.extend(boxes)
                    
                    else:
                        # MODE RAPIDE
                        progress_bar = st.progress(0)
                        info_placeholder = st.empty()
                        
                        all_detections = []
                        last_annotated = None
                        last_status = "En cours..."
                        
                        with st.spinner("🔄 Traitement de la vidéo en cours..."):
                            for annotated, status, boxes, progress, frame_num, total, processed in process_video_with_pil(
                                model, video_path, conf, iou, frame_skip, resize_factor
                            ):
                                progress_bar.progress(progress)
                                info_placeholder.text(f"Frame {frame_num}/{total} (traitées: {processed})")
                                
                                last_annotated = annotated
                                last_status = status
                                
                                if boxes:
                                    all_detections.extend(boxes)
                        
                        if last_annotated is not None:
                            st.markdown(f"### {last_status}")
                            st.image(last_annotated, caption="Dernière frame analysée", use_container_width=True)
                    
                    # Résumé final
                    st.success("✅ Analyse vidéo terminée!")
                    
                    if all_detections:
                        st.markdown("### 📊 Résumé des détections")
                        
                        class_counts = Counter([d["class"] for d in all_detections])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total détections", len(all_detections))
                        with col2:
                            st.metric("Classes uniques", len(class_counts))
                        
                        st.markdown("#### Détections par classe")
                        st.bar_chart(class_counts)
                    else:
                        st.info("Aucune poubelle détectée dans la vidéo.")
                    
                    cleanup(video_path)

            elif uploaded_video and not model:
                st.error("Veuillez charger le modèle d'abord.")
    
    # ============================================
    # TAB 3: WEBCAM (avec PIL uniquement)
    # ============================================
    with tab3:
        st.header("📹 Détection via Webcam")
        
        if not model:
            st.error("⚠️ Veuillez d'abord charger un modèle dans la barre latérale.")
        else:
            st.info("💡 Cliquez sur 'Prendre une photo' pour capturer une image depuis votre webcam.")
            
            # Utiliser la fonction native de Streamlit
            camera_image = st.camera_input("📸 Prendre une photo")
            
            if camera_image:
                # Charger avec PIL
                img = Image.open(camera_image)
                img_array = np.array(img)
                
                with st.spinner("🔍 Analyse en cours..."):
                    annotated, boxes, status = run_image_inference(
                        model, img_array, conf, iou
                    )
                
                if annotated is not None:
                    st.success("✅ Analyse terminée !")
                    st.subheader(f"Résultat : {status}")
                    st.image(annotated, use_container_width=True)
                    
                    if boxes:
                        st.markdown("### 📋 Détections")
                        st.dataframe(boxes, use_container_width=True)
                    
                    # Téléchargement
                    buf = io.BytesIO()
                    Image.fromarray(annotated.astype("uint8")).save(buf, format="JPEG")
                    buf.seek(0)
                    st.download_button(
                        "📥 Télécharger le résultat",
                        buf,
                        "webcam_detection.jpg",
                        mime="image/jpeg"
                    )
                else:
                    st.warning("Aucun résultat. Réessayez avec une autre photo.")
            
            st.markdown("---")
            st.info("""
            💡 **Astuce** : Pour une détection en continu, prenez plusieurs photos successives !
            
            Pour un vrai streaming webcam, utilisez ce script Python local :
            ```python
            import cv2
            from ultralytics import YOLO

            model = YOLO('best.pt')
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.predict(frame, conf=0.25)
                annotated = results[0].plot()
                
                cv2.imshow('Detection', annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            ```
            """)


if __name__ == "__main__":
    main()