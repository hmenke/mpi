// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a C++ wrapper class for an `MPI_Comm` object.
 */

#pragma once

#include "./communicator.hpp"

#include <mpi.h>

#include <cassert>

#include <span>

namespace mpi {

  template <class BaseType> class shared_window;

  /// The window class
  template <class BaseType>
  class window {
    friend class shared_window<BaseType>;
    MPI_Win win{MPI_WIN_NULL};
    bool is_owned{true};
  public:
    std::span<BaseType> data;
    MPI_Aint size_byte;
    int disp_unit_;

    window() = default;
    window(window const&) = delete;
    window(window &&other) noexcept : win{std::exchange(other.win, MPI_WIN_NULL)}, 
                                      is_owned{std::exchange(other.is_owned, true)}, 
                                      data{std::exchange(other.data, std::span<BaseType>())},
                                      size_byte{std::exchange(other.size_byte, MPI_Aint{0})},
                                      disp_unit_{std::exchange(other.disp_unit_, 0)} {}
    window& operator=(window const&) = delete;
    window& operator=(window &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        this->free();
        this->win = std::exchange(rhs.win, MPI_WIN_NULL);
        this->data = std::exchange(rhs.data, std::span<BaseType>());
        this->is_owned = std::exchange(rhs.is_owned, true);
        this->size_byte = std::exchange(rhs.size_byte, MPI_Aint{0});
        this-> disp_unit_ = std::exchange(rhs.disp_unit_, 0);
      }
      return *this;
    }

    /// Create a window over an existing local memory buffer
    explicit window(communicator &c, BaseType *base, MPI_Aint size = 0, MPI_Info info = MPI_INFO_NULL) noexcept(false) {
      if (base == nullptr && size != 0) { 
        std::abort(); // Undefined Behavior with Nullptr and non zero size
      }
      size = size < 0 ? 0 : size;
      disp_unit_ = sizeof(BaseType);
      size_byte = (size * disp_unit_);
      if (has_env) {
        MPI_Win_create(base, size_byte, disp_unit_, info, c.get(), &win); 
      } else {
        is_owned = false;
        data = std::span<BaseType>(base, size);
      }
    }

    /// Create a window and allocate memory for a local memory buffer
    explicit window(communicator &c, MPI_Aint size = 0, MPI_Info info = MPI_INFO_NULL) noexcept {
      size = size < 0 ? 0 : size;
      disp_unit_ = sizeof(BaseType);
      size_byte = (size * disp_unit_);
      if (has_env) {
        void *baseptr = nullptr;
        MPI_Win_allocate(size_byte, disp_unit_, info, c.get(), &baseptr, &win);
      } else {
        is_owned = true;
        BaseType* p = new BaseType[size];
        data = std::span<BaseType>(p, size);
      }
    }

    ~window() { free(); }

    explicit operator MPI_Win() const noexcept { return win; };
    explicit operator MPI_Win*() noexcept { return &win; };

    void free() noexcept {
      if (has_env) {
        if (win != MPI_WIN_NULL) {
          MPI_Win_free(&win);
        }
      } else {
        if (is_owned) {
          delete[] data.data();
        }
        data = std::span<BaseType>();
        size_byte = 0;
        disp_unit_ = 0;
      }
    }

    /// Synchronization routine in active target RMA. It opens and closes an access epoch.
    void fence(int assert = 0) const noexcept {
      if (has_env) {
        MPI_Win_fence(assert, win);
      }
    }

    /// Complete all outstanding RMA operations at both the origin and the target
    void flush(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_flush_all(win);
        } else {
          MPI_Win_flush(rank, win);
        }
      }
    }

    /// Synchronize the private and public copies of the window
    void sync() const noexcept {
      if (has_env) {
        MPI_Win_sync(win);
      }
    }

    /// Starts an RMA access epoch locking access to a particular or all ranks in the window
    void lock(int rank = -1, int lock_type = MPI_LOCK_SHARED, int assert = 0) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_lock_all(assert, win);
        } else {
          MPI_Win_lock(lock_type, rank, assert, win);
        }
      }
    }

    /// Completes an RMA access epoch started by a call to lock()
    void unlock(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_unlock_all(win);
        } else {
          MPI_Win_unlock(rank, win);
        }
      }
    }

    /// Load data from a remote memory window.
    template <typename TargetType = BaseType, typename OriginType>
    requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void get(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      int target_count_ = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        MPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
      } else {
        // What would happen if target_disp is negative
        if (target_rank != 0) {
          return;
        }

        std::span<OriginType> origin(origin_addr, origin_count);
        auto target_begin = data.begin();
        std::advance(target_begin, target_disp);
        auto target_end = target_begin;
        std::advance(target_end, target_count_);
        std::copy(target_begin, target_end, origin.begin());
      }
    }

    /// Store data to a remote memory window.
    template <typename TargetType = BaseType, typename OriginType>
    requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void put(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      int target_count_ = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
      } else {
        if (target_rank != 0) {
          return;
        }

        std::span<OriginType> origin(origin_addr, origin_count);
        auto target_begin = data.begin();
        std::advance(target_begin, target_disp);
        std::copy(origin.begin(), origin.end(), target_begin);
      }
    }

    /// Obtains the value of a window attribute.
    void* get_attr(int win_keyval) const noexcept {
      if (has_env) {
        int flag;
        void *attribute_val;
        MPI_Win_get_attr(win, win_keyval, &attribute_val, &flag);
        assert(flag);
        return attribute_val;
      } else {
        if (win_keyval == MPI_WIN_BASE) {
          return data.data();
        } else if (win_keyval == MPI_WIN_SIZE) {
          return const_cast<void*>(static_cast<const void*>(&size_byte)); 
        } else if (win_keyval == MPI_WIN_DISP_UNIT) {
          return const_cast<void*>(static_cast<const void*>(&disp_unit_)); 
        } // CREATE_FLAVOR ,MODEL , LOCK_FREE
        return nullptr;
      }
    }

    // Expose some commonly used attributes
    BaseType* base() const noexcept {
      if (has_env) {
        if (win == MPI_WIN_NULL) {
          return nullptr;
        }
        return static_cast<BaseType*>(get_attr(MPI_WIN_BASE));
      } else {
        return data.data();
      }
    }
    MPI_Aint size() const noexcept {
      if (has_env) {
        return *static_cast<MPI_Aint*>(get_attr(MPI_WIN_SIZE)); // Size in bytes
      } else {
        return data.size(); // Number of elements in the span (should be mentioned?)
      }
    }
    int disp_unit() const noexcept {
      if (has_env) {
        return *static_cast<int*>(get_attr(MPI_WIN_DISP_UNIT));
      } else {
        return disp_unit_;
      }
    }
  };

  /// The shared_window class
  template <class BaseType>
  class shared_window : public window<BaseType> {
  public:
    shared_window() = default;

    /// Create a window and allocate memory for a shared memory buffer
    explicit shared_window(shared_communicator& c, MPI_Aint size, MPI_Info info = MPI_INFO_NULL) noexcept {
      size = size < 0 ? 0 : size;
      this->disp_unit_ = sizeof(BaseType);
      this->size_byte = (size * this->disp_unit_); // Mention that negative size are turned to 0
      if (has_env) {
        void* baseptr = nullptr;
        MPI_Win_allocate_shared(this->size_byte, this->disp_unit_, info, c.get(), &baseptr, &(this->win));
      } else {
        this->is_owned = true;
        BaseType* p = new BaseType[size];
        this->data = std::span<BaseType>(p, size);
      }
    }

    /// Query a shared memory window
    std::tuple<MPI_Aint, int, void*> query(int rank = MPI_PROC_NULL) const noexcept {
      if (has_env) {
        MPI_Aint size = 0;
        int disp_unit = 0;
        void *baseptr = nullptr;
        MPI_Win_shared_query(this->win, rank, &size, &disp_unit, &baseptr);
        return {size, disp_unit, baseptr};
      } else {
        return {this->data.size(), sizeof(BaseType), this->data.data()};
      }
    }

    // Override the commonly used attributes of the window base class
    BaseType* base(int rank = MPI_PROC_NULL) const noexcept { 
      if (has_env) {
        return static_cast<BaseType*>(std::get<2>(query(rank))); 
      } else {
        return this->data.data();
      }
      
    }
    MPI_Aint size(int rank = MPI_PROC_NULL) const noexcept {
      if (has_env) {
        return std::get<0>(query(rank)) / sizeof(BaseType); 
      } else {
        return this->data.size();
      }
    }
    int disp_unit(int rank = MPI_PROC_NULL) const noexcept { 
      if (has_env) {
        return std::get<1>(query(rank)); 
      } else {
        return sizeof(BaseType);
      }
    }
  };

};
