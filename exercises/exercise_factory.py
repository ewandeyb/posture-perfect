"""
Factory pattern for exercise instantiation.
Automatically discovers and registers exercises.
"""
import importlib
import pkgutil
from typing import Dict, List, Type, Optional
from pathlib import Path
from exercises.base_exercise import BaseExercise

class ExerciseFactory:
    """
    Central registry for all exercises.
    Automatically discovers exercise classes in the exercises directory.
    """
    
    _exercises: Dict[str, Type['BaseExercise']] = {}
    _initialized = False
    
    @classmethod
    def register(cls, exercise_id: str):
        """
        Decorator to register an exercise class.
        
        Usage:
            @ExerciseFactory.register('scoliosis_side_bend')
            class SideBend(BaseExercise):
                ...
        
        Args:
            exercise_id: Unique identifier for the exercise (use snake_case)
        """
        def wrapper(exercise_class: Type['BaseExercise']):
            if exercise_id in cls._exercises:
                print(f"Warning: Exercise '{exercise_id}' is already registered. Overwriting.")
            
            cls._exercises[exercise_id] = exercise_class
            print(f"✓ Registered exercise: {exercise_id}")
            return exercise_class
        return wrapper
    
    @classmethod
    def create(cls, exercise_id: str) -> 'BaseExercise':
        """
        Create an instance of the specified exercise.
        
        Args:
            exercise_id: Unique identifier for the exercise
            
        Returns:
            Instance of the exercise class
            
        Raises:
            ValueError: If exercise_id not found
        """
        if not cls._initialized:
            cls.auto_discover()
        
        if exercise_id not in cls._exercises:
            available = ', '.join(sorted(cls._exercises.keys()))
            raise ValueError(
                f"Exercise '{exercise_id}' not found.\n"
                f"Available exercises: {available}"
            )
        
        return cls._exercises[exercise_id]()
    
    @classmethod
    def get_available_exercises(cls) -> Dict[str, str]:
        """
        Get all available exercises with their display names.
        
        Returns:
            Dict mapping exercise_id to display name
        """
        if not cls._initialized:
            cls.auto_discover()
        
        result = {}
        for exercise_id, exercise_class in cls._exercises.items():
            try:
                # Create temporary instance to get name
                instance = exercise_class()
                result[exercise_id] = instance.get_name()
            except Exception as e:
                print(f"Warning: Could not get name for {exercise_id}: {e}")
                result[exercise_id] = exercise_id.replace('_', ' ').title()
        
        return result
    
    @classmethod
    def get_exercises_by_category(cls) -> Dict[str, List[str]]:
        """
        Group exercises by category (based on module path).
        
        Returns:
            Dict mapping category to list of exercise_ids
        
        Example:
            {
                'scoliosis': ['scoliosis_side_bend', 'scoliosis_cat_cow'],
                'shoulder': ['shoulder_rotation'],
                'general': ['basic_stretch']
            }
        """
        if not cls._initialized:
            cls.auto_discover()
        
        categories = {}
        
        for exercise_id, exercise_class in cls._exercises.items():
            module_path = exercise_class.__module__
            
            # Extract category from module path
            # e.g., "exercises.scoliosis.side_bend" -> "scoliosis"
            parts = module_path.split('.')
            
            if len(parts) >= 2 and parts[0] == 'exercises':
                category = parts[1]  # Get the subfolder name
            else:
                category = 'general'
            
            if category not in categories:
                categories[category] = []
            
            categories[category].append(exercise_id)
        
        # Sort exercises within each category
        for category in categories:
            categories[category].sort()
        
        return categories
    
    @classmethod
    def get_exercise_info(cls, exercise_id: str) -> Optional[Dict]:
        """
        Get detailed information about an exercise without instantiating it.
        
        Args:
            exercise_id: The exercise identifier
            
        Returns:
            Dict with exercise metadata or None if not found
        """
        if not cls._initialized:
            cls.auto_discover()
        
        if exercise_id not in cls._exercises:
            return None
        
        try:
            instance = cls.create(exercise_id)
            return {
                'id': exercise_id,
                'name': instance.get_name(),
                'description': instance.get_description(),
                'instructions': instance.get_instructions(),
                'common_mistakes': instance.get_common_mistakes(),
                'required_landmarks': instance.get_required_landmarks()
            }
        except Exception as e:
            print(f"Error getting info for {exercise_id}: {e}")
            return None
    
    @classmethod
    def auto_discover(cls):
        """
        Automatically discover and import all exercise modules.
        This walks through the exercises/ directory and imports all Python files.
        Call this once at application startup.
        """
        if cls._initialized:
            return
        
        print("🔍 Auto-discovering exercises...")
        
        try:
            import exercises
            
            # Get the path to the exercises package
            exercises_path = Path(exercises.__file__).parent
            
            # Walk through all modules in the exercises package
            for importer, modname, ispkg in pkgutil.walk_packages(
                path=[str(exercises_path)],
                prefix='exercises.',
            ):
                # Skip __init__, base_exercise, and exercise_factory modules
                if any(skip in modname for skip in ['__init__', 'base_exercise', 'exercise_factory']):
                    continue
                
                try:
                    # Import the module (this triggers @register decorators)
                    importlib.import_module(modname)
                except ImportError as e:
                    print(f"⚠️  Could not import {modname}: {e}")
                except Exception as e:
                    print(f"⚠️  Error loading {modname}: {e}")
            
            cls._initialized = True
            print(f"✓ Discovery complete. Found {len(cls._exercises)} exercises.\n")
            
        except Exception as e:
            cls._initialized = True  # Prevent repeated attempts
            print(f"❌ Error during auto-discovery: {e}")
    
    @classmethod
    def list_all(cls) -> None:
        """
        Print a formatted list of all available exercises.
        Useful for debugging and documentation.
        """
        if not cls._initialized:
            cls.auto_discover()
        
        categories = cls.get_exercises_by_category()
        
        print("\n" + "="*60)
        print("AVAILABLE EXERCISES")
        print("="*60)
        
        for category, exercise_ids in sorted(categories.items()):
            print(f"\n📁 {category.upper()}")
            print("-" * 60)
            
            for ex_id in exercise_ids:
                try:
                    instance = cls.create(ex_id)
                    print(f"  • {ex_id:<30} → {instance.get_name()}")
                except Exception as e:
                    print(f"  • {ex_id:<30} → [Error: {e}]")
        
        print("\n" + "="*60 + "\n")
    
    @classmethod
    def reset(cls):
        """
        Reset the factory (useful for testing).
        Clears all registered exercises.
        """
        cls._exercises.clear()
        cls._initialized = False


# Auto-discover exercises when this module is imported
# This happens once when the application starts
ExerciseFactory.auto_discover()