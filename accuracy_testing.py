import cv2
from main import match_face_to_reference


class SimilarityTester:
    def __init__(self):
        self.lowest_trainer_success = 1.0
        self.highest_trainer_failure = 0.0
        self.lowest_pokemon_success = 1.0
        self.highest_pokemon_failure = 0.0
        self.lowest_combined_success = 1.0
        self.highest_combined_failure = 0.0
        self.intended_results = 0
        self.unintended_results = 0

    def run_match_faces(self, debug_message, should_succeed, img1, img2, threshold):
        trainer_accuracy, pokemon_accuracy = match_face_to_reference(img1, img2, 0)
        success = min(trainer_accuracy, pokemon_accuracy) > threshold
        if should_succeed:
            self.lowest_trainer_success = min(self.lowest_trainer_success, trainer_accuracy)
            self.lowest_pokemon_success = min(self.lowest_pokemon_success, pokemon_accuracy)
            self.lowest_combined_success = min(self.lowest_combined_success, min(trainer_accuracy, pokemon_accuracy))
        else:
            self.highest_trainer_failure = max(self.highest_trainer_failure, trainer_accuracy)
            self.highest_pokemon_failure = max(self.highest_pokemon_failure, pokemon_accuracy)
            self.highest_combined_failure = max(self.highest_combined_failure, min(trainer_accuracy, pokemon_accuracy))
        if success == should_succeed:
            self.intended_results += 1
            print(f"{debug_message}: had intended result. trainer: {trainer_accuracy}, pokemon: {pokemon_accuracy}")
        else:
            self.unintended_results += 1
            print(f"{debug_message}: had unintended result. trainer: {trainer_accuracy}, pokemon: {pokemon_accuracy}")

    def print_summary(self):
        print("__ summary")
        print(f"intended results {self.intended_results}, unintended results {self.unintended_results}")
        print(f"trainer score: highest failure {self.highest_trainer_failure},"
              f" lowest success {self.lowest_trainer_success}")
        print(f"pokemon score: highest failure {self.highest_pokemon_failure},"
              f" lowest success {self.lowest_pokemon_success}")
        print(f"combined score: highest failure {self.highest_combined_failure},"
              f" lowest success {self.lowest_combined_success}")


def test_matching_faces():
    tester = SimilarityTester()
    threshold = .7

    print(f"___ nc red 1")
    img1 = cv2.imread("examples\\similarity_tests\\ncred_ece3d5806_0.jpg")
    img2 = cv2.imread("examples\\similarity_tests\\ncred_ed2b75844_0.jpg")
    img3 = cv2.imread("examples\\similarity_tests\\ncred_ee8b7480e_0.jpg")
    tester.run_match_faces("red1 and red2", True, img1, img2, threshold)
    tester.run_match_faces("red1 and red3", True, img1, img3, threshold)
    tester.run_match_faces("red2 and red3", True, img2, img3, threshold)

    print(f"___ nc red 2")
    img1 = cv2.imread("examples\\similarity_tests\\red2_e62f96c0a_1.jpg")
    img2 = cv2.imread("examples\\similarity_tests\\red2_e62f96c0a_17.jpg")
    img3 = cv2.imread("examples\\similarity_tests\\red2_e62f96c0a_43.jpg")
    tester.run_match_faces("red1 and red2", True, img1, img2, threshold)
    tester.run_match_faces("red1 and red3", True, img1, img3, threshold)
    tester.run_match_faces("red2 and red3", True, img2, img3, threshold)

    print(f"___ mudkips")
    img1 = cv2.imread("examples\\similarity_tests\\mudkip_1.jpg")
    img2 = cv2.imread("examples\\similarity_tests\\mudkip_2.jpg")
    img3 = cv2.imread("examples\\similarity_tests\\mudkip_3_false.jpg")
    tester.run_match_faces("mudkip_1 and mudkip_2", True, img1, img2, threshold)
    tester.run_match_faces("mudkip_1 and mudkip_3_false", False, img1, img3, threshold)
    tester.run_match_faces("mudkip_2 and mudkip_3_false", False, img2, img3, threshold)

    tester.print_summary()


if __name__ == "__main__":
    test_matching_faces()
